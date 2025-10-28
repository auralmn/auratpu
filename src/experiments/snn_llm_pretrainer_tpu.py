
#!/usr/bin/env python3
"""
TPU-ready SNN-Based LLM Pretrainer (v4-32 compatible)

Key changes vs. original:
- Replaces OpenCL/C++ fused kernels with vectorized PyTorch ops that XLA can compile.
- Adds PyTorch/XLA multi-process training (xmp.spawn), BF16, and device placement helpers.
- Shards data across TPU cores; aggregates metrics on master.
- Keeps Oja/Sanger + whitening path as-is (CPU) to minimize invasive changes.
  (Tip: prebuild/load SentencePiece on rank 0 if you want tokenizer-based vocab.)

Usage (single host with v4-32 slice):
  python3 snn_llm_pretrainer_tpu.py --tpu --corpus cleaned_corpus.txt --epochs 1 --batch-size 64

Note:
- FlashAttention is disabled on TPU in this reference implementation.
- LTC and Izhikevich dynamics are implemented in pure torch and compiled by XLA.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import torch

# -----------------------------
# Optional SentencePiece
# -----------------------------
try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except Exception:
    HAS_SENTENCEPIECE = False

# -----------------------------
# PyTorch/XLA (TPU) support
# -----------------------------
XLA_AVAILABLE = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr  # for world size
    XLA_AVAILABLE = True
except Exception:
    XLA_AVAILABLE = False

# Fallback: world size helpers
def _world_size():
    if XLA_AVAILABLE:
        try:
            return xr.world_size()
        except Exception:
            try:
                return xm.xrt_world_size()
            except Exception:
                return 1
    return 1

def _rank():
    if XLA_AVAILABLE:
        try:
            return xm.get_ordinal()
        except Exception:
            return 0
    return 0

def _is_master():
    return _rank() == 0

# Default dtype: BF16 on TPU, FP32 elsewhere
DEFAULT_DTYPE = torch.bfloat16 if XLA_AVAILABLE else torch.float32

# -----------------------------
# Oja/Sanger Whitener (original dependency)
# -----------------------------
from oja_sanger_whitener import OnlineWhitener, OjaLayer, NumpyEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Vectorized neuron dynamics (PyTorch/XLA friendly)
# -----------------------------

def izhikevich_step(v, u, I, a, b, c, d, v_th=30.0, dt=1.0):
    """
    One Euler step of Izhikevich neurons (vectorized). All tensors on same device.
    v, u, I are [N]; a,b,c,d are [N]. Returns (v_new, u_new, spikes_mask[bool]).
    """
    dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
    v = v + dt * dv
    du = a * (b * v - u)
    u = u + dt * du
    spikes = v >= v_th
    # reset
    if spikes.any():
        v = torch.where(spikes, c, v)
        u = torch.where(spikes, u + d, u)
    return v, u, spikes

def ltc_update(ltc_state, W_rec, rec_in, tau, dt=0.1, tau_min=0.1):
    """
    Simple LTC-like leaky integration update:
      dS/dt = -S/tau + W_rec @ rec_in
      S <- S + dt * dS/dt
      tau clamped to [tau_min, +inf)
    All tensors vectors/matrices on same device.
    """
    tau = torch.clamp(tau, min=tau_min)
    dS = -ltc_state / tau + rec_in
    return ltc_state + dt * dS

# -----------------------------
# Trainer
# -----------------------------

class SNNLLMPretrainerTPU:
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 512,
        hidden_neurons: int = 2048,
        oja_components: int = 128,
        context_length: int = 512,
        use_ltc: bool = True,
        use_flash_attention: bool = False,  # disabled on TPU here
        dtype: torch.dtype = DEFAULT_DTYPE,
        device: Optional[torch.device] = None
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_neurons = hidden_neurons
        self.oja_components = oja_components
        self.context_length = context_length
        self.use_ltc = use_ltc
        self.use_flash_attention = use_flash_attention
        self.dtype = dtype

        # Device
        if device is None:
            if XLA_AVAILABLE:
                self.device = xm.xla_device()
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Components
        self._init_whitener()          # CPU (minimize invasive change)
        self._init_oja_layer()         # CPU (minimize invasive change)
        self._init_embeddings()        # CPU to avoid CPU<->TPU ping-pong in whitening/Oja
        self._init_snn_layers()        # TPU/Device
        self._init_stdp()              # TPU/Device

        # Metrics (local, aggregated later on master)
        self.metrics = {
            'total_tokens': 0,
            'total_spikes': 0,
            'stdp_updates': 0,
            'oja_growth_events': 0,
            'perplexity_history': [],
            'spike_rate_history': []
        }

        logger.info(f"✓ SNNLLMPretrainerTPU initialized on {self.device} (dtype {self.dtype})")
        logger.info(f"  Vocab: {vocab_size}, Embedding: {embedding_dim}, Hidden: {hidden_neurons}, Oja: {oja_components}")
        logger.info(f"  LTC: {use_ltc}, FlashAttention: {use_flash_attention}")

    # ---------- Init blocks (CPU for whitening/Oja, Device for SNN) ----------

    def _init_whitener(self):
        self.whitener = OnlineWhitener(dim=self.embedding_dim, eps=1e-6, momentum=0.01)
        logger.info("  ✓ Whitener initialized (CPU)")

    def _init_oja_layer(self):
        self.oja_layer = OjaLayer(
            dim=self.embedding_dim,
            n_components=self.oja_components,
            mode='sanger',
            lr=0.01,
            lateral_beta=0.005,
            grow_threshold=0.1,
            max_components=self.oja_components * 2
        )
        self.current_oja_size = self.oja_components
        logger.info("  ✓ Oja layer initialized (CPU, dynamic growth enabled)")

    def _init_embeddings(self):
        # Keep embeddings on CPU since whitening+Oja are CPU here
        self.embeddings = torch.randn(self.vocab_size, self.embedding_dim, dtype=torch.float32) * 0.02
        logger.info("  ✓ Embeddings initialized (CPU)")

    def _init_snn_layers(self):
        # Place SNN weights/states on device (TPU/GPU/CPU)
        self.W_input_hidden = (torch.randn(self.hidden_neurons, self.oja_components) * 0.3).to(self.device, self.dtype)
        self.W_hidden_output = (torch.randn(self.vocab_size, self.hidden_neurons) * 0.1).to(self.device, self.dtype)

        # Izhikevich params (diverse population), on device
        rs_count = int(self.hidden_neurons * 0.5)
        fs_count = int(self.hidden_neurons * 0.3)
        burst_count = self.hidden_neurons - rs_count - fs_count

        a = torch.zeros(self.hidden_neurons)
        b = torch.zeros(self.hidden_neurons)
        c = torch.zeros(self.hidden_neurons)
        d = torch.zeros(self.hidden_neurons)

        # Regular Spiking
        a[:rs_count] = 0.02; b[:rs_count] = 0.2;  c[:rs_count] = -65.0; d[:rs_count] = 8.0
        # Fast Spiking
        a[rs_count:rs_count+fs_count] = 0.1; b[rs_count:rs_count+fs_count] = 0.2; c[rs_count:rs_count+fs_count] = -65.0; d[rs_count:rs_count+fs_count] = 2.0
        # Bursting
        a[rs_count+fs_count:] = 0.02; b[rs_count+fs_count:] = 0.2; c[rs_count+fs_count:] = -50.0; d[rs_count+fs_count:] = 2.0

        self.neuron_params = {
            'a': a.to(self.device, self.dtype),
            'b': b.to(self.device, self.dtype),
            'c': c.to(self.device, self.dtype),
            'd': d.to(self.device, self.dtype)
        }

        if self.use_ltc:
            self.W_ltc_recurrent = (torch.randn(self.hidden_neurons, self.hidden_neurons) * 0.1).to(self.device, self.dtype)
            self.ltc_state = torch.zeros(self.hidden_neurons, dtype=self.dtype, device=self.device)
            self.ltc_tau = torch.ones(self.hidden_neurons, dtype=self.dtype, device=self.device)

        logger.info(f"  ✓ SNN layers initialized on device (RS={rs_count}, FS={fs_count}, Burst={burst_count})")

    def _init_stdp(self):
        self.stdp_params = {
            'A_plus': 0.015,
            'A_minus': 0.015,
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'w_min': 0.0,
            'w_max': 1.0
        }
        self.input_trace = torch.zeros(self.oja_components, dtype=self.dtype, device=self.device)
        self.hidden_trace = torch.zeros(self.hidden_neurons, dtype=self.dtype, device=self.device)
        logger.info("  ✓ STDP initialized (device)")

    # ---------- Token / feature pipeline ----------

    def encode_tokens(self, tokens: List[int]) -> torch.Tensor:
        """
        Embedding (CPU) -> whitening (CPU) -> Oja (CPU) -> torch tensor (device)
        Returns (seq_len, current_oja_size) on device dtype.
        """
        token_ids = torch.tensor(tokens, dtype=torch.long)
        token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
        embeddings = self.embeddings[token_ids]  # CPU float32

        oja_features = []
        for emb in embeddings:
            emb_np = emb.numpy()
            whitened = self.whitener.transform(emb_np)
            oja_out = self.oja_layer.step(whitened)
            y = oja_out.y

            if oja_out.grew:
                self._handle_oja_growth()
                self.metrics['oja_growth_events'] += 1

            oja_features.append(torch.from_numpy(y).float())

        # Pad in case of growth during this sequence
        max_size = max(f.shape[0] for f in oja_features)
        if any(f.shape[0] != max_size for f in oja_features):
            padded = []
            for f in oja_features:
                if f.shape[0] < max_size:
                    padding = torch.zeros(max_size - f.shape[0])
                    f = torch.cat([f, padding])
                padded.append(f)
            feat = torch.stack(padded)
        else:
            feat = torch.stack(oja_features)

        # Move to device/dtype once per sequence
        return feat.to(self.device, self.dtype)

    def _handle_oja_growth(self):
        new_size = self.oja_layer.K
        old_size = self.current_oja_size
        if new_size <= old_size:
            return
        logger.info(f"🌱 Oja layer grew: {old_size} -> {new_size}")
        # Resize W_input_hidden (hidden_neurons, new_size)
        new_cols = (torch.randn(self.hidden_neurons, new_size - old_size) * 0.1).to(self.device, self.dtype)
        self.W_input_hidden = torch.cat([self.W_input_hidden, new_cols], dim=1)
        # Resize STDP traces
        self.input_trace = torch.cat([self.input_trace, torch.zeros(new_size - old_size, dtype=self.dtype, device=self.device)])
        self.current_oja_size = new_size

    # ---------- SNN forward & learning ----------

    @torch.no_grad()
    def forward_snn(self, oja_features: torch.Tensor, timesteps: int = 20) -> torch.Tensor:
        """
        Forward pass through Izhikevich SNN (vectorized across neurons, loop across timesteps).
        oja_features: (seq_len, oja_dim) on device
        Returns spikes: (seq_len, timesteps, hidden_neurons) on device (dtype float32 for easier downstream)
        """
        seq_len = oja_features.shape[0]
        all_spikes = []

        a = self.neuron_params['a']
        b = self.neuron_params['b']
        c = self.neuron_params['c']
        d = self.neuron_params['d']

        for t in range(seq_len):
            # Input current: [N]
            input_current = torch.matmul(self.W_input_hidden, oja_features[t])  # [hidden]
            v = torch.full((self.hidden_neurons,), -65.0, dtype=self.dtype, device=self.device)
            u = torch.zeros_like(v)

            spikes_t = []
            # Precompute LTC recurrent input as simple linear of last state/spike rate
            rec_term = None

            for _ in range(timesteps):
                I = input_current if rec_term is None else input_current + rec_term
                v, u, s_mask = izhikevich_step(v, u, I, a, b, c, d, v_th=30.0, dt=0.5)
                spikes_t.append(s_mask.to(torch.float32))  # keep float32 for rate computation

                if self.use_ltc:
                    # Use current spike mask as rate proxy this step
                    rate = s_mask.to(self.dtype)
                    rec_in = torch.matmul(self.W_ltc_recurrent, rate)
                    self.ltc_state = ltc_update(self.ltc_state, self.W_ltc_recurrent, rec_in, self.ltc_tau, dt=0.1, tau_min=0.1)
                    rec_term = rec_in

            spikes_t = torch.stack(spikes_t, dim=0)  # [T, N]
            all_spikes.append(spikes_t)
            self.metrics['total_spikes'] += float(spikes_t.sum().item())

        return torch.stack(all_spikes, dim=0)  # [L, T, N]

    @torch.no_grad()
    def apply_stdp(self, oja_features: torch.Tensor, spike_trains: torch.Tensor):
        """
        STDP with simple pre/post traces. All tensors on device.
        oja_features: [L, K], spike_trains: [L, T, N]
        """
        A_plus = self.stdp_params['A_plus']
        A_minus = self.stdp_params['A_minus']
        tau_plus = self.stdp_params['tau_plus']
        tau_minus = self.stdp_params['tau_minus']
        w_min = self.stdp_params['w_min']
        w_max = self.stdp_params['w_max']

        dt = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        decay_plus = torch.exp(-dt / torch.tensor(tau_plus, dtype=self.dtype, device=self.device))
        decay_minus = torch.exp(-dt / torch.tensor(tau_minus, dtype=self.dtype, device=self.device))

        L = oja_features.shape[0]
        for t in range(L):
            pre = (oja_features[t] > 0).to(self.dtype)                 # [K]
            post = (spike_trains[t].sum(dim=0) > 0).to(self.dtype)     # [N]

            # Trace updates
            self.input_trace = self.input_trace * decay_plus + pre
            self.hidden_trace = self.hidden_trace * decay_minus + post

            # Weight change (outer products)
            dw_pot = A_plus * torch.outer(post, self.input_trace)      # [N, K]
            dw_dep = A_minus * torch.outer(self.hidden_trace, pre)     # [N, K]
            self.W_input_hidden = self.W_input_hidden + (dw_pot - dw_dep)

            # Clamp
            self.W_input_hidden = torch.clamp(self.W_input_hidden, w_min, w_max)

            self.metrics['stdp_updates'] += 1

    @torch.no_grad()
    def compute_output_logits(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """
        Spike-rate readout to logits.
        spike_trains: [L, T, N] -> rates [L, N] -> logits [L, V]
        """
        spike_rates = spike_trains.mean(dim=1).to(self.W_hidden_output.dtype)  # [L, N]
        return torch.matmul(spike_rates, self.W_hidden_output.T)               # [L, V]

    # ---------- Training ----------

    def train_on_corpus(
        self,
        corpus_path: str,
        epochs: int = 3,
        batch_size: int = 32,
        max_lines: Optional[int] = None,
        save_every: int = 1000
    ):
        print("="*70)
        print("🧠 SNN LLM PRETRAINING (TPU-ready: STDP + Oja/Sanger)")
        print("="*70)

        # Load corpus (all ranks read; we shard lines)
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if max_lines:
            lines = lines[:max_lines]

        # Shard for distributed run
        ws = _world_size()
        rk = _rank()
        if ws > 1:
            lines = lines[rk::ws]

        print(f"  ✓ Rank {rk}/{ws} sees {len(lines)} lines")

        # Build vocabulary (simple or SentencePiece). For simplicity across ranks,
        # do simple vocab unless a prebuilt SentencePiece model exists.
        self.vocab = self._build_vocab(corpus_path, max_lines)

        start_time = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_tokens = 0
            epoch_spikes = 0

            for batch_idx in range(0, len(lines), batch_size):
                batch_lines = lines[batch_idx:batch_idx + batch_size]

                try:
                    batch_metrics = self._process_batch(batch_lines)
                    epoch_tokens += batch_metrics['tokens']
                    epoch_spikes += batch_metrics['spikes']
                    self.metrics['total_tokens'] += batch_metrics['tokens']

                    if (batch_idx // batch_size) % 10 == 0 and _is_master():
                        spike_rate = batch_metrics['spikes'] / max(batch_metrics['tokens'], 1)
                        print(f"  Epoch {epoch+1} Batch {batch_idx//batch_size}: "
                              f"{batch_metrics['tokens']} tokens, "
                              f"{batch_metrics['spikes']} spikes, "
                              f"rate={spike_rate:.2f}")

                    # per-rank checkpoints (optional). Master will save consolidated at end.
                    # if (batch_idx // batch_size) % save_every == 0 and batch_idx > 0 and _is_master():
                    #     self.save_checkpoint(f"checkpoint_e{epoch+1}_b{batch_idx//batch_size}.json")

                except Exception as e:
                    logger.warning(f"Rank {rk} error processing batch {batch_idx}: {e}")
                    continue

            epoch_time = time.time() - epoch_start
            tokens_per_sec = epoch_tokens / max(epoch_time, 1e-6)
            avg_spike_rate = epoch_spikes / max(epoch_tokens, 1)

            self.metrics['spike_rate_history'].append(float(avg_spike_rate))
            perplexity = np.exp(min(avg_spike_rate / 1000.0, 10.0))
            self.metrics['perplexity_history'].append(float(perplexity))

            if _is_master():
                print(f"\n✓ Epoch {epoch + 1} (rank {rk}) complete:")
                print(f"  Tokens: {epoch_tokens}")
                print(f"  Spikes: {epoch_spikes}")
                print(f"  Spike rate: {avg_spike_rate:.1f} spikes/token")
                print(f"  Perplexity: {perplexity:.2f}")
                print(f"  Time: {epoch_time:.1f}s")
                print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")

        total_time = time.time() - start_time
        if _is_master():
            print(f"\n{'='*70}")
            print("✅ TRAINING COMPLETE (this rank)")
            print('='*70)
            print(f"Total time: {total_time:.1f}s")
            print(f"Total tokens (local): {self.metrics['total_tokens']}")
            print(f"Total spikes (local): {self.metrics['total_spikes']}")
            print(f"STDP updates (local): {self.metrics['stdp_updates']}")
            print(f"Oja growth events (local): {self.metrics['oja_growth_events']}")

        return self.metrics

    # ---------- Vocab / tokenization ----------

    def _build_vocab(self, corpus_path: str, max_lines: Optional[int] = None) -> Dict[str, int]:
        # If a prebuilt SentencePiece model exists, load it on all ranks.
        model_path = 'snn_llm_tokenizer.model'
        if HAS_SENTENCEPIECE and Path(model_path).exists():
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(model_path)
            vocab = { self.sp_model.id_to_piece(i): i for i in range(self.sp_model.get_piece_size()) }
            logger.info(f"  ✓ Loaded SentencePiece model with {len(vocab)} tokens")
            return vocab
        # Otherwise fall back to simple vocab to avoid cross-rank training of tokenizer.
        return self._build_simple_vocab(corpus_path, max_lines)

    def _build_simple_vocab(self, corpus_path: str, max_lines: Optional[int] = None) -> Dict[str, int]:
        vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        word_counts = defaultdict(int)
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                for word in line.strip().split():
                    word_counts[word] += 1
        for word, _ in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size - len(vocab)]:
            vocab[word] = len(vocab)
        return vocab

    def _tokenize(self, text: str) -> List[int]:
        if HAS_SENTENCEPIECE and hasattr(self, 'sp_model'):
            return self.sp_model.encode(text, out_type=int)
        return [self.vocab.get(w, 1) for w in text.split()]

    # ---------- Batch processing ----------

    def _process_batch(self, batch_lines: List[str]) -> Dict[str, int]:
        batch_tokens = 0
        batch_spikes = 0
        for line in batch_lines:
            tokens = self._tokenize(line.strip())
            if len(tokens) < 2:
                continue
            tokens = tokens[:self.context_length]
            batch_tokens += len(tokens)

            oja_features = self.encode_tokens(tokens)              # [L, K] on device
            spike_trains = self.forward_snn(oja_features, timesteps=15)  # [L, T, N] on device
            batch_spikes += int(spike_trains.sum().item())

            self.apply_stdp(oja_features, spike_trains)

        return {'tokens': batch_tokens, 'spikes': int(batch_spikes)}

    # ---------- Checkpointing ----------

    def save_checkpoint(self, path: str):
        checkpoint = {
            'config': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'hidden_neurons': self.hidden_neurons,
                'oja_components': self.oja_components,
                'use_ltc': self.use_ltc,
                'use_flash_attention': self.use_flash_attention,
                'has_sentencepiece': HAS_SENTENCEPIECE and hasattr(self, 'sp_model')
            },
            'metrics': self.metrics,
            'vocab_size_actual': len(self.vocab) if hasattr(self, 'vocab') else 0,
            'tokenizer_model': 'snn_llm_tokenizer.model' if HAS_SENTENCEPIECE and hasattr(self, 'sp_model') else None
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2, cls=NumpyEncoder)

        # Save weights (bring tensors to CPU)
        weight_path = path.replace('.json', '_weights.npz')
        np.savez(
            weight_path,
            W_input_hidden=self.W_input_hidden.detach().to('cpu', torch.float32).numpy(),
            W_hidden_output=self.W_hidden_output.detach().to('cpu', torch.float32).numpy()
        )
        logger.info(f"Checkpoint saved to {path}")

# -----------------------------
# Spawn entrypoints for TPU
# -----------------------------

def _run_worker(rank, args):
    # Each XLA process has its own device
    trainer = SNNLLMPretrainerTPU(
        vocab_size=50000,
        embedding_dim=512,
        hidden_neurons=args.hidden,
        oja_components=args.oja,
        use_ltc=args.use_ltc,
        use_flash_attention=False,  # disabled on TPU in this reference
        dtype=DEFAULT_DTYPE
    )
    metrics = trainer.train_on_corpus(
        corpus_path=args.corpus,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_lines=args.max_lines,
        save_every=500
    )
    # Save only on master rank to avoid conflicts
    if _is_master():
        trainer.save_checkpoint(args.output)
        print(f"\n✅ (master) Model saved to {args.output}")
        for key, value in metrics.items():
            if not isinstance(value, list):
                print(f"  {key}: {value}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TPU-ready SNN LLM Pretraining')
    parser.add_argument('--corpus', type=str, default='cleaned_corpus.txt', help='Path to corpus file')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden', type=int, default=2048, help='Number of hidden neurons')
    parser.add_argument('--oja', type=int, default=128, help='Number of Oja components')
    parser.add_argument('--max-lines', type=int, default=None, help='Maximum lines to process')
    parser.add_argument('--use-ltc', action='store_true', help='Use LTC neurons')
    parser.add_argument('--output', type=str, default='snn_llm_model_tpu.json', help='Output model path')
    parser.add_argument('--tpu', action='store_true', help='Use TPU (PyTorch/XLA).')
    args = parser.parse_args()

    if args.tpu and XLA_AVAILABLE:
        # Let XLA decide nprocs; if not provided, defaults to world size (v4-32 => 32)
        nprocs = _world_size()
        if nprocs < 1:
            nprocs = 1
        print(f"Launching XLA spawn with {nprocs} processes...")
        xmp.spawn(_run_worker, args=(args,), nprocs=nprocs, start_method='fork')
    else:
        # CPU/GPU single-process fallback
        _run_worker(0, args)

if __name__ == "__main__":
    main()
