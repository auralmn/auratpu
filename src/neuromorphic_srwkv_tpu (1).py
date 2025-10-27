
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
AURA Neuromorphic SRWKV (TPU/XLA-ready) + Streaming Softmax

- Drop-in alternative to neuromorphic_srwkv_flash.py for TPUv4-32.
- No custom CUDA/OpenCL kernels; uses pure Torch ops that XLA compiles.
- Attention modes:
    * 'dot'      : standard scaled dot-product attention (full T x T)
    * 'chunked'  : block-by-block softmax to reduce peak memory
    * 'streaming': numerically stable streaming softmax (online over KV blocks)
                   with running max & partition function accumulation.

The 'streaming' variant computes, per query row, the following recurrence over KV blocks b:
    m_new = max(m, max(scores_b))
    alpha = exp(m - m_new)
    Z_new = Z * alpha + sum(exp(scores_b - m_new))
    O_new = O * alpha + exp(scores_b - m_new) @ V_b
Then output = O_new / Z_new, identical to full softmax(QK^T)V in exact arithmetic.
We compute in float32 internally for stability on BF16 TPUs, and cast back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# XLA/Device helpers
# -----------------------------
XLA_AVAILABLE = False
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    XLA_AVAILABLE = True
except Exception:
    XLA_AVAILABLE = False

def get_device():
    if XLA_AVAILABLE:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEFAULT_DTYPE = torch.bfloat16 if XLA_AVAILABLE else torch.float32


# -----------------------------
# Spiking k-WTA (tensorized)
# -----------------------------

@dataclass
class SpikingConfig:
    decay: float = 0.7
    threshold: float = 1.0
    k_winners: int = 5
    gain_up: float = 1.5
    gain_down: float = 0.6
    reset_mode: str = "soft"

class SpikingKWTA(nn.Module):
    def __init__(self, config: SpikingConfig, dtype: torch.dtype = DEFAULT_DTYPE, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device if device is not None else get_device()

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor, vocab_size: Optional[int] = None) -> torch.Tensor:
        ids = token_ids.to(self.device).view(-1)
        if ids.numel() == 0:
            return torch.ones(vocab_size if vocab_size is not None else 1, device=self.device, dtype=self.dtype)

        V = int(torch.max(ids).item()) + 1 if vocab_size is None else int(vocab_size)
        decay = torch.as_tensor(self.config.decay, device=self.device, dtype=self.dtype)
        thr = torch.as_tensor(self.config.threshold, device=self.device, dtype=self.dtype)

        potentials = torch.zeros(V, device=self.device, dtype=self.dtype)
        spikes = torch.zeros(V, device=self.device, dtype=torch.int32)

        L = ids.numel()
        for t in range(L):
            potentials.mul_(decay)
            tid = ids[t:t+1]
            delta = torch.bincount(tid, minlength=V).to(self.dtype)
            potentials.add_(delta)

            mask = potentials >= thr
            if mask.any():
                spikes.masked_scatter_(mask, (spikes[mask] + 1))
                if self.config.reset_mode == "soft":
                    potentials[mask] = potentials[mask] - thr
                else:
                    potentials[mask] = 0

        gains = torch.ones(V, device=self.device, dtype=self.dtype)
        active = spikes > 0
        if int(active.sum().item()) > 0:
            values = spikes.to(torch.float32) * 1e6 + potentials.to(torch.float32)
            k = min(self.config.k_winners, V)
            topk = torch.topk(values, k=k, largest=True).indices
            gains = gains.scatter(0, active.nonzero(as_tuple=False).view(-1), torch.as_tensor(self.config.gain_down, device=self.device, dtype=self.dtype))
            gains[topk] = torch.as_tensor(self.config.gain_up, device=self.device, dtype=self.dtype)
        return gains


# -----------------------------
# Neuromorphic SRWKV (TPU)
# -----------------------------

class NeuromorphicSRWKVTpu(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.embedding_dim = int(config.get('embedding_dim', 256))
        self.num_heads = int(config.get('num_heads', 8))
        assert self.embedding_dim % self.num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.head_dim = self.embedding_dim // self.num_heads

        self.spike_threshold = float(config.get('spike_threshold', 0.5))
        self.decay_factor = float(config.get('decay_factor', 0.9))

        self.attn_mode = str(config.get('attn_mode', 'streaming'))  # 'streaming' | 'chunked' | 'dot'
        self.block_size_q = int(config.get('block_size_q', 64))
        self.block_size_kv = int(config.get('block_size_kv', 64))

        self.kwta = SpikingKWTA(
            SpikingConfig(
                decay=float(config.get('kwta_decay', 0.7)),
                threshold=float(config.get('kwta_threshold', 1.0)),
                k_winners=int(config.get('k_winners', 5)),
                gain_up=float(config.get('gain_up', 1.5)),
                gain_down=float(config.get('gain_down', 0.6)),
                reset_mode=str(config.get('reset_mode', 'soft'))
            ),
            dtype=DEFAULT_DTYPE,
            device=get_device()
        )

        self.mu_token = float(config.get('mu_token', 0.3))
        self.mu_context = float(config.get('mu_context', 0.8))
        self.adaptation_rate = float(config.get('adaptation_rate', 0.1))

        D = self.embedding_dim
        self.receptance = nn.Linear(D, D, bias=False)
        self.key = nn.Linear(D, D, bias=False)
        self.value = nn.Linear(D, D, bias=False)
        self.output_projection = nn.Linear(D, D)

        self.time_mix_k = nn.Parameter(torch.ones(1, 1, D))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, D))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, D))

        dev = get_device()
        self.register_buffer('learning_rates', torch.ones(D, device=dev, dtype=DEFAULT_DTYPE))
        self.register_buffer('adaptation_ema', torch.zeros((), device=dev, dtype=DEFAULT_DTYPE))
        self.register_buffer('prev_state', torch.zeros(1, D, device=dev, dtype=DEFAULT_DTYPE))
        self.register_buffer('attention_weights', torch.zeros(1, 1, device=dev, dtype=DEFAULT_DTYPE))

        self.device_ = dev
        self.to(dev)

        self.initialize()

    def initialize(self) -> bool:
        try:
            g = 0.5
            nn.init.xavier_uniform_(self.receptance.weight, gain=g)
            nn.init.xavier_uniform_(self.key.weight, gain=g)
            nn.init.xavier_uniform_(self.value.weight, gain=g)
            nn.init.xavier_uniform_(self.output_projection.weight, gain=g)
            nn.init.uniform_(self.time_mix_k, 0.1, 0.9)
            nn.init.uniform_(self.time_mix_v, 0.1, 0.9)
            nn.init.uniform_(self.time_mix_r, 0.1, 0.9)
            return True
        except Exception as e:
            print(f"Init failed: {e}")
            return False

    # ---------- Neuromorphic pieces ----------

    def spike_activation(self, x: torch.Tensor) -> torch.Tensor:
        thr = torch.as_tensor(self.spike_threshold, device=x.device, dtype=x.dtype)
        spikes = (x > thr).to(x.dtype)
        if self.training:
            ste = torch.clamp(1 - torch.abs(x - thr), 0, 1)
            spikes = spikes + ste - ste.detach()
        return spikes

    def adaptive_time_mixing(self, x: torch.Tensor, prev_x: torch.Tensor,
                             mix_weight: torch.Tensor, learning_gains: torch.Tensor) -> torch.Tensor:
        mix = mix_weight.squeeze(0).squeeze(0).to(x.device, x.dtype)
        if learning_gains is not None:
            lg = learning_gains.to(x.device, x.dtype)
            if lg.shape[-1] == mix.shape[-1]:
                mix = mix * lg
        return mix * x + (1 - mix) * prev_x

    # ---------- Attention impls ----------

    def _reshape_to_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, D = t.shape
        H = self.num_heads
        Hd = self.head_dim
        return t.view(B, T, H, Hd).transpose(1, 2).contiguous()  # [B, H, T, Hd]

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, H, T, Hd = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, H * Hd)

    def scaled_dot_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale           # [B,H,T,T]
        attn = F.softmax(scores, dim=-1)                                 # [B,H,T,T]
        out = torch.matmul(attn, v)                                      # [B,H,T,Dh]
        return out

    def chunked_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          block_q: int, block_kv: int) -> torch.Tensor:
        B, H, T, Dh = q.shape
        scale = 1.0 / math.sqrt(Dh)
        outputs = []
        for q_start in range(0, T, block_q):
            q_end = min(q_start + block_q, T)
            q_blk = q[:, :, q_start:q_end, :]                              # [B,H,Bq,Dh]

            scores_blk = []
            for kv_start in range(0, T, block_kv):
                kv_end = min(kv_start + block_kv, T)
                k_blk = k[:, :, kv_start:kv_end, :]                        # [B,H,Bk,Dh]
                s = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * scale   # [B,H,Bq,Bk]
                scores_blk.append(s)
            scores_blk = torch.cat(scores_blk, dim=-1)                      # [B,H,Bq,T]

            attn_blk = F.softmax(scores_blk, dim=-1)                        # [B,H,Bq,T]

            out_blk = torch.zeros(B, H, q_end - q_start, Dh, device=q.device, dtype=q.dtype)
            kv_cursor = 0
            for kv_start in range(0, T, block_kv):
                kv_end = min(kv_start + block_kv, T)
                v_blk = v[:, :, kv_start:kv_end, :]
                a_slice = attn_blk[:, :, :, kv_cursor:kv_cursor + (kv_end - kv_start)]
                out_blk = out_blk + torch.matmul(a_slice, v_blk)
                kv_cursor += (kv_end - kv_start)

            outputs.append(out_blk)
        return torch.cat(outputs, dim=2)

    def streaming_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            block_q: int, block_kv: int) -> torch.Tensor:
        """
        Numerically stable streaming softmax attention.
        Processes KV in blocks and maintains running (m, Z, O) per query row:
          m: running max logits
          Z: partition function sum of exp(logits - m)
          O: weighted value accumulator
        All accumulators are float32 regardless of input dtype for stability on TPU.
        """
        B, H, T, Dh = q.shape
        scale = 1.0 / math.sqrt(Dh)

        outputs = []
        for q_start in range(0, T, block_q):
            q_end = min(q_start + block_q, T)
            q_blk = q[:, :, q_start:q_end, :]                                  # [B,H,Bq,Dh]

            # Compute in FP32
            qf = q_blk.to(torch.float32)
            mf = torch.full((B, H, q_end - q_start, 1), -float('inf'), device=q.device, dtype=torch.float32)  # m
            Zf = torch.zeros(B, H, q_end - q_start, 1, device=q.device, dtype=torch.float32)                   # Z
            Of = torch.zeros(B, H, q_end - q_start, Dh, device=q.device, dtype=torch.float32)                  # O

            for kv_start in range(0, T, block_kv):
                kv_end = min(kv_start + block_kv, T)
                k_blk = k[:, :, kv_start:kv_end, :]
                v_blk = v[:, :, kv_start:kv_end, :]

                kf = k_blk.to(torch.float32)
                vf = v_blk.to(torch.float32)

                # scores: [B,H,Bq,Bk]
                scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
                # block max along last dim
                m_block = scores.max(dim=-1, keepdim=True).values                            # [B,H,Bq,1]
                m_new = torch.maximum(mf, m_block)                                           # [B,H,Bq,1]

                # alpha = exp(m - m_new)
                alpha = torch.exp(mf - m_new)

                # exp(scores - m_new)
                exp_scores = torch.exp(scores - m_new)

                # Update Z and O
                Zf = Zf * alpha + exp_scores.sum(dim=-1, keepdim=True)                       # [B,H,Bq,1]
                Of = Of * alpha + torch.matmul(exp_scores, vf)                                # [B,H,Bq,Dh]
                mf = m_new

            out_blk = Of / (Zf + 1e-9)                                                       # [B,H,Bq,Dh]
            # Cast back to input dtype
            outputs.append(out_blk.to(q.dtype))

        return torch.cat(outputs, dim=2)                                                      # [B,H,T,Dh]

    # ---------- Core compute ----------

    def compute_neuromorphic_attention(self, query: torch.Tensor, key: torch.Tensor,
                                       value: torch.Tensor, token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.device_
        B, T, D = query.shape
        query = query.to(device, DEFAULT_DTYPE)
        key = key.to(device, DEFAULT_DTYPE)
        value = value.to(device, DEFAULT_DTYPE)

        # k-WTA learning gains
        if token_ids is not None:
            gains_vec = self.kwta(token_ids, vocab_size=None)
            if gains_vec.numel() >= D:
                learning_gains = gains_vec[:D]
            else:
                learning_gains = torch.ones(D, device=device, dtype=DEFAULT_DTYPE)
                learning_gains[:gains_vec.numel()] = gains_vec
        else:
            learning_gains = torch.ones(D, device=device, dtype=DEFAULT_DTYPE)

        if self.prev_state.size(0) != B:
            self.prev_state = torch.zeros(B, D, device=device, dtype=DEFAULT_DTYPE)

        outputs = []
        current_state = self.prev_state.clone()
        for t in range(T):
            x_t = query[:, t, :]
            k_in = self.adaptive_time_mixing(x_t, current_state, self.time_mix_k, learning_gains)
            v_in = self.adaptive_time_mixing(x_t, current_state, self.time_mix_v, learning_gains)
            r_in = self.adaptive_time_mixing(x_t, current_state, self.time_mix_r, learning_gains)

            k_t = self.key(k_in)
            v_t = self.value(v_in)
            r_t = self.receptance(r_in)

            k_s = self.spike_activation(k_t)
            v_s = self.spike_activation(v_t)
            r_s = self.spike_activation(r_t)

            outputs.append(r_s * k_s * v_s)

            adaptive_decay = torch.as_tensor(self.decay_factor * float(learning_gains.mean().item()),
                                             device=device, dtype=DEFAULT_DTYPE)
            current_state = adaptive_decay * current_state + (1 - adaptive_decay) * x_t

        temporal = torch.stack(outputs, dim=1)  # [B,T,D]
        self.prev_state = current_state.detach()

        qh = self._reshape_to_heads(temporal)
        kh = self._reshape_to_heads(temporal)
        vh = self._reshape_to_heads(temporal)

        if self.attn_mode == 'streaming':
            out_h = self.streaming_attention(qh, kh, vh, self.block_size_q, self.block_size_kv)
        elif self.attn_mode == 'chunked':
            out_h = self.chunked_attention(qh, kh, vh, self.block_size_q, self.block_size_kv)
        else:
            out_h = self.scaled_dot_attention(qh, kh, vh)

        out = self._merge_heads(out_h)  # [B,T,D]
        out = self.output_projection(out)
        return out

    # ---------- Public API ----------

    def forward(self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.compute_neuromorphic_attention(x, x, x, token_ids)

    def get_spike_statistics(self) -> Dict[str, float]:
        return {
            'state_magnitude': float(torch.norm(self.prev_state).item()),
            'spike_threshold': float(self.spike_threshold),
            'attn_mode': self.attn_mode,
        }

    def validate(self) -> Tuple[bool, str]:
        try:
            B, T, D = 2, 10, self.embedding_dim
            x = torch.randn(B, T, D, device=self.device_, dtype=DEFAULT_DTYPE)
            ids = torch.randint(0, 1024, (B, T), device=self.device_)
            y = self.forward(x, ids)
            if y.shape != x.shape:
                return False, f"Shape mismatch: {y.shape} vs {x.shape}"
            if torch.isnan(y).any() or torch.isinf(y).any():
                return False, "Output has NaN/Inf"
            return True, f"OK ({self.attn_mode} attention, device={self.device_}, dtype={DEFAULT_DTYPE})"
        except Exception as e:
            return False, f"Validate error: {e}"


# -----------------------------
# Demo / Quick test
# -----------------------------
if __name__ == "__main__":
    cfg = {
        'embedding_dim': 256,
        'num_heads': 8,
        'attn_mode': 'streaming',     # 'streaming' | 'chunked' | 'dot'
        'block_size_q': 32,
        'block_size_kv': 64,
        'spike_threshold': 0.5,
        'decay_factor': 0.9,
        'k_winners': 5,
    }
    model = NeuromorphicSRWKVTpu(cfg)
    B, T = 4, 64
    x = torch.randn(B, T, cfg['embedding_dim'], device=get_device(), dtype=DEFAULT_DTYPE)
    token_ids = torch.randint(0, 4096, (B, T), device=get_device())
    y = model(x, token_ids)
    ok, msg = model.validate()
    print("Forward:", tuple(y.shape), "| Validate:", msg)
