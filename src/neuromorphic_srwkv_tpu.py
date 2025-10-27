
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
AURA Neuromorphic SRWKV (TPU/XLA-ready)
- Drop-in alternative to neuromorphic_srwkv_flash.py for TPUv4-32.
- No custom CUDA/OpenCL kernels; uses pure Torch ops that XLA compiles.
- Optional "chunked attention" approximates FlashAttention behavior (lower memory).

Key changes:
- Device plumbing for PyTorch/XLA (TPU) with BF16 default.
- Spiking k-WTA rewritten as tensor ops using scatter/bincount (stateless per forward).
- Attention path:
    * 'chunked' attention: block-by-block scaled dot product with softmax.
    * or standard dot-product attention as fallback.
- Retains SRWKV-style time mixing and neuromorphic spikes.
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
    decay: float = 0.7          # membrane leak factor [0,1)
    threshold: float = 1.0      # spiking threshold
    k_winners: int = 5          # number of k-WTA winners
    gain_up: float = 1.5        # LR multiplier for winners
    gain_down: float = 0.6      # LR multiplier for non-winners
    reset_mode: str = "soft"    # "soft" or "hard"

class SpikingKWTA(nn.Module):
    """
    Spike-based k-WTA implemented with tensors for XLA.
    Stateless across calls; computes decayed potentials and spike counts for a sequence.
    """
    def __init__(self, config: SpikingConfig, dtype: torch.dtype = DEFAULT_DTYPE, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device if device is not None else get_device()

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor, vocab_size: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            token_ids: (..., ) integer tensor of token IDs (batch, seq) or (seq,)
            vocab_size: optional vocabulary size; if None, inferred as max+1
        Returns:
            gains: (vocab_size,) tensor with LR multipliers on self.device
        """
        ids = token_ids.to(self.device).view(-1)
        if ids.numel() == 0:
            return torch.ones(vocab_size if vocab_size is not None else 1, device=self.device, dtype=self.dtype)

        V = int(torch.max(ids).item()) + 1 if vocab_size is None else int(vocab_size)
        decay = torch.as_tensor(self.config.decay, device=self.device, dtype=self.dtype)
        thr = torch.as_tensor(self.config.threshold, device=self.device, dtype=self.dtype)

        # State tensors
        potentials = torch.zeros(V, device=self.device, dtype=self.dtype)
        spikes = torch.zeros(V, device=self.device, dtype=torch.int32)

        # Process sequence step-by-step (XLA will compile this loop)
        # For batched tokens at step t, use bincount to add one per token id.
        L = ids.numel()
        for t in range(L):
            # Leak
            potentials.mul_(decay)

            # Add current token(s)
            tid = ids[t:t+1]  # single id in this scan; keeps semantics with original
            delta = torch.bincount(tid, minlength=V).to(self.dtype)
            potentials.add_(delta)

            # Spike where threshold crossed
            mask = potentials >= thr
            if mask.any():
                spikes.masked_scatter_(mask, (spikes[mask] + 1))
                if self.config.reset_mode == "soft":
                    potentials[mask] = potentials[mask] - thr
                else:
                    potentials[mask] = 0

        # Compute k-WTA gains
        gains = torch.ones(V, device=self.device, dtype=self.dtype)
        active = spikes > 0
        if int(active.sum().item()) > 0:
            # Top-k by spike count; tie-breaker by residual potential
            values = spikes.to(torch.float32) * 1e6 + potentials.to(torch.float32)  # big weight to spike counts
            k = min(self.config.k_winners, V)
            topk = torch.topk(values, k=k, largest=True).indices
            gains = gains.scatter(0, active.nonzero(as_tuple=False).view(-1), torch.as_tensor(self.config.gain_down, device=self.device, dtype=self.dtype))
            gains[topk] = torch.as_tensor(self.config.gain_up, device=self.device, dtype=self.dtype)

        return gains


# -----------------------------
# Neuromorphic SRWKV (TPU)
# -----------------------------

class NeuromorphicSRWKVTpu(nn.Module):
    """
    SRWKV with neuromorphic spikes + TPU-friendly attention.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # SRWKV parameters
        self.embedding_dim = int(config.get('embedding_dim', 256))
        self.num_heads = int(config.get('num_heads', 8))
        assert self.embedding_dim % self.num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.head_dim = self.embedding_dim // self.num_heads

        # Neuromorphic parameters
        self.spike_threshold = float(config.get('spike_threshold', 0.5))
        self.decay_factor = float(config.get('decay_factor', 0.9))

        # Attention mode
        self.attn_mode = str(config.get('attn_mode', 'chunked'))  # 'chunked' | 'dot'
        self.block_size_q = int(config.get('block_size_q', 64))
        self.block_size_kv = int(config.get('block_size_kv', 64))

        # Spiking k-WTA
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

        # NLMS-like gains
        self.mu_token = float(config.get('mu_token', 0.3))
        self.mu_context = float(config.get('mu_context', 0.8))
        self.adaptation_rate = float(config.get('adaptation_rate', 0.1))

        # Projections
        D = self.embedding_dim
        self.receptance = nn.Linear(D, D, bias=False)
        self.key = nn.Linear(D, D, bias=False)
        self.value = nn.Linear(D, D, bias=False)
        self.output_projection = nn.Linear(D, D)

        # Time mixing
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, D))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, D))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, D))

        # Buffers / state
        dev = get_device()
        self.register_buffer('learning_rates', torch.ones(D, device=dev, dtype=DEFAULT_DTYPE))
        self.register_buffer('adaptation_ema', torch.zeros((), device=dev, dtype=DEFAULT_DTYPE))
        self.register_buffer('prev_state', torch.zeros(1, D, device=dev, dtype=DEFAULT_DTYPE))
        self.register_buffer('attention_weights', torch.zeros(1, 1, device=dev, dtype=DEFAULT_DTYPE))

        self.device_ = dev
        self.to(dev)

        # Init
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
            # Straight-through est. with triangular window around threshold
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
        # q,k,v: [B,H,T,Dh]
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale           # [B,H,T,T]
        attn = F.softmax(scores, dim=-1)                                 # [B,H,T,T]
        out = torch.matmul(attn, v)                                      # [B,H,T,Dh]
        return out

    def chunked_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          block_q: int, block_kv: int) -> torch.Tensor:
        """
        Memory-friendlier attention evaluated block-by-block along query length.
        q,k,v: [B,H,T,Dh]
        Returns: [B,H,T,Dh]
        """
        B, H, T, Dh = q.shape
        scale = 1.0 / math.sqrt(Dh)

        outputs = []
        for q_start in range(0, T, block_q):
            q_end = min(q_start + block_q, T)
            q_blk = q[:, :, q_start:q_end, :]                              # [B,H,Bq,Dh]

            # Compute scores for this query block against all keys in chunks to cap temporary memory
            # Accumulate softmax in one go per query-block (ok for moderate T).
            scores_blk = []
            for kv_start in range(0, T, block_kv):
                kv_end = min(kv_start + block_kv, T)
                k_blk = k[:, :, kv_start:kv_end, :]                        # [B,H,Bk,Dh]
                s = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * scale   # [B,H,Bq,Bk]
                scores_blk.append(s)
            scores_blk = torch.cat(scores_blk, dim=-1)                      # [B,H,Bq,T]

            attn_blk = F.softmax(scores_blk, dim=-1)                        # [B,H,Bq,T]

            # Multiply by V in chunks
            out_blk = torch.zeros(B, H, q_end - q_start, Dh, device=q.device, dtype=q.dtype)
            kv_cursor = 0
            for kv_start in range(0, T, block_kv):
                kv_end = min(kv_start + block_kv, T)
                v_blk = v[:, :, kv_start:kv_end, :]                         # [B,H,Bk,Dh]
                a_slice = attn_blk[:, :, :, kv_cursor:kv_cursor + (kv_end - kv_start)]  # [B,H,Bq,Bk]
                out_blk = out_blk + torch.matmul(a_slice, v_blk)            # accumulate
                kv_cursor += (kv_end - kv_start)

            outputs.append(out_blk)                                         # [B,H,Bq,Dh]

        return torch.cat(outputs, dim=2)                                    # [B,H,T,Dh]

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
            gains_vec = self.kwta(token_ids, vocab_size=None)           # [V]
            # Map gains to feature dim: pad/trim
            if gains_vec.numel() >= D:
                learning_gains = gains_vec[:D]
            else:
                learning_gains = torch.ones(D, device=device, dtype=DEFAULT_DTYPE)
                learning_gains[:gains_vec.numel()] = gains_vec
        else:
            learning_gains = torch.ones(D, device=device, dtype=DEFAULT_DTYPE)

        # Ensure prev_state has correct batch size
        if self.prev_state.size(0) != B:
            self.prev_state = torch.zeros(B, D, device=device, dtype=DEFAULT_DTYPE)

        # SRWKV temporal path with adaptive mixing + spikes
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

            outputs.append(r_s * k_s * v_s)  # [B,D]

            # Update state with adaptive decay (scalar)
            adaptive_decay = torch.as_tensor(self.decay_factor * float(learning_gains.mean().item()),
                                             device=device, dtype=DEFAULT_DTYPE)
            current_state = adaptive_decay * current_state + (1 - adaptive_decay) * x_t

        temporal = torch.stack(outputs, dim=1)  # [B,T,D]
        self.prev_state = current_state.detach()

        # Global attention over neuromorphic sequence
        qh = self._reshape_to_heads(temporal)
        kh = self._reshape_to_heads(temporal)
        vh = self._reshape_to_heads(temporal)

        if self.attn_mode == 'chunked':
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
        'attn_mode': 'chunked',       # or 'dot'
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
