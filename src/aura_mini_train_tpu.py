#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
AURA Mini LLM — TPU-ready trainer (torch-xla 2.8 compatible)

- Uses PJRT (PJRT_DEVICE=TPU). For core count, set env TPU_NUM_DEVICES
  (or restrict with TPU_VISIBLE_CHIPS). spawn() is called without nprocs.
- Pre-tokenized mmap dataset (--bin-dir) or raw-text fallback.
- Neuromorphic SRWKV mixer with streaming softmax (from neuromorphic_srwkv_tpu.py).
- Grad accumulation, grad clipping, cosine LR warmup/decay.
- Smoothed loss meters (EMA + rolling window) averaged across TPU cores.
- Rank-aware logging via xm.master_print.
"""

from __future__ import annotations

import os
import math
import json
import time
import argparse
from typing import Optional, List
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# XLA (TPU)
# -----------------------------------------------------------------------------
XLA_AVAILABLE = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr
    XLA_AVAILABLE = True
except Exception:
    XLA_AVAILABLE = False

DEFAULT_DTYPE = torch.bfloat16 if XLA_AVAILABLE else torch.float32

# -----------------------------------------------------------------------------
# Neuromorphic mixer (your module)
# -----------------------------------------------------------------------------
from neuromorphic_srwkv_tpu import NeuromorphicSRWKVTpu, get_device, DEFAULT_DTYPE as SRWKV_DTYPE

# -----------------------------------------------------------------------------
# Tokenizer (byte-level with specials)
# -----------------------------------------------------------------------------
class ByteTokenizer:
    def __init__(self):
        self.PAD, self.BOS, self.EOS, self.MEM, self.SEP = 256, 257, 258, 259, 260
        self.vocab_size = 261
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids = list(text.encode('utf-8', errors='ignore'))
        return ([self.BOS] + ids + [self.EOS]) if add_special else ids
    def decode(self, ids: List[int]) -> str:
        b = [i for i in ids if 0 <= i < 256]
        try: return bytes(b).decode('utf-8', errors='ignore')
        except Exception: return bytes(b).decode('latin-1', errors='ignore')

# -----------------------------------------------------------------------------
# Model blocks
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * norm

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 4.0):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class AURABlock(nn.Module):
    def __init__(self, dim: int, heads: int, attn_mode: str, block_q: int, block_kv: int):
        super().__init__()
        cfg = {
            'embedding_dim': dim,
            'num_heads': heads,
            'attn_mode': attn_mode,
            'block_size_q': block_q,
            'block_size_kv': block_kv,
            'spike_threshold': 0.5,
            'decay_factor': 0.9,
            'k_winners': 4,
        }
        self.norm1 = RMSNorm(dim)
        self.mix = NeuromorphicSRWKVTpu(cfg)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_mult=4.0)
    def forward(self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.mix(self.norm1(x), token_ids)
        x = x + self.mlp(self.norm2(x))
        return x

class AURAMiniLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, heads: int, layers: int,
                 attn_mode: str, block_q: int, block_kv: int, chkpt: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            AURABlock(dim, heads, attn_mode, block_q, block_kv) for _ in range(layers)
        ])
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.use_checkpoint = chkpt
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        if self.use_checkpoint:
            for blk in self.blocks:
                x = torch.utils.checkpoint.checkpoint(lambda t: blk(t, token_ids), x, use_reentrant=False)
        else:
            for blk in self.blocks:
                x = blk(x, token_ids)
        x = self.norm_f(x)
        return self.lm_head(x)

# -----------------------------------------------------------------------------
# Datasets (pretokenized fast path + raw fallback)
# -----------------------------------------------------------------------------
class PreTokenizedDataset(torch.utils.data.Dataset):
    """Memmap tokens with fixed windows; produced by pack_to_bin.py. Returns (x,y)."""
    def __init__(self, bin_dir: str):
        meta = json.load(open(f"{bin_dir}/meta.json","r"))
        self.seq_len = int(meta["seq_len"])
        self.idx = np.load(f"{bin_dir}/idx.npy")
        self.tokens = np.memmap(f"{bin_dir}/tokens.bin", mode="r", dtype=np.uint16)
        self.vocab_size = int(meta.get("vocab_size", 261))
    def __len__(self): return len(self.idx)
    def __getitem__(self, i: int):
        s = int(self.idx[i]); T = self.seq_len + 1
        window = self.tokens[s:s+T].astype(np.int64)
        x = torch.from_numpy(window[:-1].copy()).long()
        y = torch.from_numpy(window[1:].copy()).long()
        return x, y

class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, lines: List[str], tokenizer: ByteTokenizer, seq_len: int):
        self.tok = tokenizer
        self.seq_len = seq_len
        ids: List[int] = []
        for ln in lines:
            ids.extend(self.tok.encode(ln.strip(), add_special=True))
        pad_id = self.tok.PAD
        T = seq_len + 1
        if len(ids) % T != 0:
            ids.extend([pad_id] * (T - (len(ids) % T)))
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.nseq = len(ids) // T
    def __len__(self): return self.nseq
    def __getitem__(self, i: int):
        s = i * (self.seq_len + 1)
        e = s + (self.seq_len + 1)
        window = self.tokens[s:e]
        return window[:-1], window[1:]

# -----------------------------------------------------------------------------
# Utilities: rank/world, logging, ckpt, eval, smooth meters
# -----------------------------------------------------------------------------
def _rank():
    if XLA_AVAILABLE:
        try: return xm.get_ordinal()
        except Exception: return 0
    return 0

def _world_size():
    if XLA_AVAILABLE:
        try: return xr.world_size()
        except Exception: return 1
    return 1

_is_master = lambda: _rank() == 0

def log(*args):
    msg = " ".join(str(a) for a in args)
    if XLA_AVAILABLE:
        xm.master_print(msg)
    else:
        print(msg, flush=True)

class EmaMeter:
    def __init__(self, beta: float = 0.98):
        self.beta = float(beta); self.value = None; self.count = 0
    def update(self, x: float):
        self.value = float(x) if self.value is None else self.beta*self.value + (1.0-self.beta)*float(x)
        self.count += 1

class WindowMean:
    def __init__(self, window: int = 100):
        self.buf = deque(maxlen=int(window))
    def update(self, x: float): self.buf.append(float(x))
    @property
    def value(self) -> float: return float('nan') if not self.buf else sum(self.buf)/len(self.buf)
    @property
    def count(self) -> int: return len(self.buf)

def _dist_mean_scalar(x: float, device: torch.device) -> float:
    if XLA_AVAILABLE:
        t = torch.tensor([x], device=device, dtype=torch.float32)
        xm.all_reduce(xm.REDUCE_MEAN, [t])
        return float(t.item())
    return float(x)

@torch.no_grad()
def evaluate(model: nn.Module, dl: torch.utils.data.DataLoader, device: torch.device,
             vocab_size: int, max_steps: int = 50) -> float:
    model.eval(); losses = []; steps = 0
    for x, y in dl:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.detach().float().cpu()); steps += 1
        if steps >= max_steps: break
    model.train()
    if not losses: return float('inf')
    return math.exp(min(torch.stack(losses).mean().item(), 20.0))

def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, meta: dict):
    if not _is_master(): return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    chk = {'meta': meta,
           'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
           'optim_state': optimizer.state_dict()}
    torch.save(chk, path)

# -----------------------------------------------------------------------------
# Train loop (worker)
# -----------------------------------------------------------------------------
def train_worker(index, args):
    device = get_device()
    torch.manual_seed(1337 + _rank())

    tok = ByteTokenizer()

    # Runtime world size / rank (now accurate inside worker)
    ws = _world_size()
    rk = _rank()

    # Dataset + per-rank sampler
    if args.bin_dir:
        ds = PreTokenizedDataset(args.bin_dir)
        tok.vocab_size = ds.vocab_size
    else:
        with open(args.corpus, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if args.max_lines: lines = lines[:args.max_lines]
        ds = PackedDataset(lines, tok, args.seq_len)

    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=ws, rank=rk, shuffle=True, drop_last=True
    ) if XLA_AVAILABLE else None

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=args.workers,
        persistent_workers=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None,
        pin_memory=False,
    )

    # Startup logs
    log(f"rank {rk}/{ws} | device={device} | dtype={DEFAULT_DTYPE}")
    log(f"dataset windows={len(ds)} | seq_len={args.seq_len} | batch_size={args.batch_size} | microbatches={args.microbatches} | log_every={args.log_every}")

    # Model
    model = AURAMiniLM(
        vocab_size=tok.vocab_size,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        attn_mode=args.attn_mode,
        block_q=args.block_q,
        block_kv=args.block_kv,
        chkpt=args.ckpt,
    ).to(device)
    model.train()

    # Optimizer & Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return max(1e-8, step / max(1, args.warmup_steps))
        if args.total_steps <= args.warmup_steps: return 1.0
        progress = (step - args.warmup_steps) / (args.total_steps - args.warmup_steps)
        progress = max(0.0, min(1.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    ema_loss = EmaMeter(beta=args.ema_beta)
    win_loss = WindowMean(window=args.avg_window)

    steps = 0
    tokens_per_step = args.batch_size * args.seq_len * ws * max(1, args.microbatches)
    log(f"global tokens/step ≈ {tokens_per_step:,}")
    t0 = time.time()

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for x, y in dl:
            model.train()
            opt.zero_grad(set_to_none=True)
            B = x.size(0)
            m = max(1, args.microbatches)
            mb_sz = max(1, B // m) if m > 1 else B

            accum_loss = 0.0
            for i in range(m):
                s = i * mb_sz
                e = B if i == m - 1 else min(B, (i + 1) * mb_sz)
                if s >= e: continue
                xi = x[s:e].to(device); yi = y[s:e].to(device)
                logits = model(xi)
                loss = F.cross_entropy(logits.view(-1, tok.vocab_size), yi.view(-1))
                (loss / m).backward()
                accum_loss += float(loss.detach().cpu())

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if XLA_AVAILABLE:
                xm.optimizer_step(opt, barrier=True)
                xm.mark_step()
            else:
                opt.step()
            scheduler.step()

            step_loss = accum_loss / m
            step_loss_global = _dist_mean_scalar(step_loss, device)
            ema_loss.update(step_loss_global)
            win_loss.update(step_loss_global)

            steps += 1

            if _is_master() and (steps % args.log_every == 0):
                elapsed = time.time() - t0
                toks_sec = tokens_per_step * (args.log_every / max(1e-6, elapsed))
                lr_now = scheduler.get_last_lr()[0]
                inst_ppl = math.exp(min(step_loss_global, 20.0))
                ema_ppl  = math.exp(min(ema_loss.value, 20.0)) if ema_loss.count > 0 else float('nan')
                win_ppl  = math.exp(min(win_loss.value, 20.0)) if win_loss.count > 0 else float('nan')
                log(
                    f"ep {epoch+1} step {steps} | "
                    f"loss {step_loss_global:.4f} (ema {ema_loss.value:.4f}, win {win_loss.value:.4f}) | "
                    f"ppl {inst_ppl:.2f} (ema {ema_ppl:.2f}, win {win_ppl:.2f}) | "
                    f"lr {lr_now:.3e} | {toks_sec:,.0f} tok/s"
                )
                t0 = time.time()

            if args.save_every and steps % args.save_every == 0:
                save_checkpoint(f"ckpts/aura_mini_step{steps}.pt", model, opt, meta={
                    'dim': args.dim, 'heads': args.heads, 'layers': args.layers,
                    'seq_len': args.seq_len, 'vocab_size': tok.vocab_size,
                    'step': steps, 'epoch': epoch+1,
                })

            if args.total_steps and steps >= args.total_steps:
                break

        if args.total_steps and steps >= args.total_steps:
            break

    if _is_master():
        save_checkpoint("ckpts/aura_mini_final.pt", model, opt, meta={
            'dim': args.dim, 'heads': args.heads, 'layers': args.layers,
            'seq_len': args.seq_len, 'vocab_size': tok.vocab_size,
            'total_steps': steps,
        })
        log("Saved final checkpoint: ckpts/aura_mini_final.pt")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument('--bin-dir', type=str, default=None)
    ap.add_argument('--corpus', type=str, default='demo.txt')
    ap.add_argument('--seq-len', type=int, default=2048)
    ap.add_argument('--max-lines', type=int, default=None)
    ap.add_argument('--workers', type=int, default=2)

    # Model
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--attn-mode', type=str, default='streaming', choices=['streaming','chunked','dot'])
    ap.add_argument('--block-q', type=int, default=128)
    ap.add_argument('--block-kv', type=int, default=256)
    ap.add_argument('--ckpt', action='store_true')

    # Train
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--microbatches', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--beta1', type=float, default=0.9)
    ap.add_argument('--beta2', type=float, default=0.95)
    ap.add_argument('--weight-decay', type=float, default=0.1)
    ap.add_argument('--warmup-steps', type=int, default=200)
    ap.add_argument('--total-steps', type=int, default=0)
    ap.add_argument('--grad-clip', type=float, default=1.0)

    # Smoothed loss
    ap.add_argument('--ema-beta', type=float, default=0.98)
    ap.add_argument('--avg-window', type=int, default=100)

    # TPU controls (torch-xla 2.8+)
    ap.add_argument('--tpu', action='store_true')
    ap.add_argument('--tpu-cores', type=int, default=0,
                    help='If >0, sets env TPU_NUM_DEVICES before spawn; spawn nprocs is None')

    args = ap.parse_args()

    if args.tpu:
        if not XLA_AVAILABLE:
            raise RuntimeError("torch-xla unavailable. Install torch-xla and set PJRT_DEVICE=TPU.")
        # torch-xla 2.8 expects you to limit devices via env; spawn() picks them up.
        if args.tpu_cores and os.environ.get('TPU_NUM_DEVICES') is None:
            os.environ['TPU_NUM_DEVICES'] = str(int(args.tpu_cores))
        # If you need to skip bad chips, set TPU_VISIBLE_CHIPS="0,1,3" in your shell before running.
        xmp.spawn(train_worker, args=(args,), start_method='fork')
    else:
        train_worker(0, args)

if __name__ == '__main__':
    main()
