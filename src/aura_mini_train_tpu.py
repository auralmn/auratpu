
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
AURA Mini LLM (Byte-level) — TPUv4-32 friendly
- Small model (~30M params by default) to validate training/inference, logging, and numerics
- Depends on neuromorphic_srwkv_tpu.NeuromorphicSRWKVTpu for the mixer/attention
- Pure PyTorch + PyTorch/XLA (no external tokenizer dependencies)

Usage:
  python aura_mini_train_tpu.py --tpu --corpus demo.txt --epochs 1 --seq-len 2048

Files:
  - aura_mini_llm.py (this)
  - neuromorphic_srwkv_tpu.py (already created earlier)
"""

from __future__ import annotations
import os, math, json, time, argparse, random
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# XLA
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

# ---------- Tokenizer (byte-level) ----------


# --- PATCH: add memory/RAG to AURA Mini ---
import os, math, json, time, argparse, random
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromorphic_srwkv_tpu import NeuromorphicSRWKVTpu, get_device, DEFAULT_DTYPE as SRWKV_DTYPE

# Memory store
from memory_store import MemoryStore

# ---------- Tokenizer (byte-level + memory specials) ----------
class ByteTokenizer:
    def __init__(self):
        # bytes 0..255 + special tokens
        self.PAD, self.BOS, self.EOS, self.MEM, self.SEP = 256, 257, 258, 259, 260
        self.vocab_size = 261
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids = list(text.encode('utf-8', errors='ignore'))
        if add_special:
            return [self.BOS] + ids + [self.EOS]
        return ids
    def encode_memory_prefix(self, docs: List[str]) -> List[int]:
        ids = [self.MEM]
        for i, d in enumerate(docs):
            ids += self.encode(d, add_special=False)
            if i != len(docs)-1: ids += [self.SEP]
        return ids + [self.SEP]
    def decode(self, ids: List[int]) -> str:
        bytes_list = [i for i in ids if 0 <= i < 256]
        try:
            return bytes(bytes_list).decode('utf-8', errors='ignore')
        except Exception:
            return bytes(bytes_list).decode('latin-1', errors='ignore')

# Inject the updated tokenizer and RAG into the existing file by replacing the prior sections.
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

# ---------- Neuromorphic block wrapper ----------
from neuromorphic_srwkv_tpu import NeuromorphicSRWKVTpu, get_device, DEFAULT_DTYPE as SRWKV_DTYPE

class AURABlock(nn.Module):
    def __init__(self, dim: int, heads: int, attn_mode: str, block_q: int, block_kv: int):
        super().__init__()
        cfg = {
            'embedding_dim': dim,
            'num_heads': heads,
            'attn_mode': attn_mode,         # 'streaming'|'chunked'|'dot'
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

# ---------- AURA Mini Model ----------

class AURAMiniLM(nn.Module):
    def __init__(self, vocab_size: int = 259, dim: int = 512, heads: int = 8, layers: int = 6,
                 attn_mode: str = 'streaming', block_q: int = 128, block_kv: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            AURABlock(dim, heads, attn_mode, block_q, block_kv) for _ in range(layers)
        ])
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B,T]
        x = self.embed(token_ids)
        for blk in self.blocks:
            x = blk(x, token_ids)
        x = self.norm_f(x)
        logits = self.lm_head(x)  # [B,T,V]
        return logits

# ---------- Dataset ----------

class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, lines: List[str], tokenizer: ByteTokenizer, seq_len: int):
        self.tok = tokenizer
        self.seq_len = seq_len
        # pack tokens end-to-end with BOS/EOS
        ids: List[int] = []
        for ln in lines:
            ids.extend(self.tok.encode(ln.strip(), add_special=True))
        # pad to multiple of seq_len
        pad_id = self.tok.PAD
        if len(ids) % seq_len != 0:
            ids.extend([pad_id] * (seq_len - (len(ids) % seq_len)))
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.nseq = len(ids) // seq_len
    def __len__(self): return self.nseq
    def __getitem__(self, i: int):
        s = i * self.seq_len
        e = s + self.seq_len
        x = self.tokens[s:e]
        y = self.tokens[s+1:e+1]  # next-token
        return x[:-1], y[:-1]     # keep shapes equal

# ---------- Training ----------

def _world_size():
    if XLA_AVAILABLE:
        try: return xr.world_size()
        except Exception: return 1
    return 1
def _rank():
    if XLA_AVAILABLE:
        try: return xm.get_ordinal()
        except Exception: return 0
    return 0
def _is_master(): return _rank()==0

def train_worker(index, args):
    device = get_device()
    torch.manual_seed(42 + _rank())
    tok = ByteTokenizer()

    # Data
    with open(args.corpus, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if args.max_lines:
        lines = lines[:args.max_lines]

    # shard across ranks (DP)
    ws = _world_size()
    rk = _rank()
    if ws > 1:
        lines = lines[rk::ws]

    ds = PackedDataset(lines, tok, args.seq_len)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    ms = MemoryStore(args.memory_dir) if args.memory_dir else None
    def maybe_replay(batch_x: torch.Tensor, batch_y: torch.Tensor):
        if ms is None or args.replay_ratio <= 0.0:
            return batch_x, batch_y
        B, T = batch_x.shape
        n_replay = int(B * args.replay_ratio)
        if n_replay == 0:
            return batch_x, batch_y
        # Sample random memory items and replace first n_replay samples
        import random
        idxs = random.sample(range(len(ms.items)), k=min(n_replay, len(ms.items))) if len(ms.items)>0 else []
        for j, mi in enumerate(idxs):
            txt = ms.items[mi].text[:args.seq_len]
            ids = tok.encode(txt, add_special=True)
            ids = ids[:args.seq_len]
            if len(ids)<2: ids = ids + [tok.PAD]
            xj = torch.tensor(ids[:-1], dtype=torch.long)
            yj = torch.tensor(ids[1:], dtype=torch.long)
            batch_x[j,:] = xj
            batch_y[j,:] = yj
        return batch_x, batch_y

    # Model
    model = AURAMiniLM(
        vocab_size=tok.vocab_size,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        attn_mode=args.attn_mode,
        block_q=args.block_q,
        block_kv=args.block_kv,
    ).to(device)
    model.train()

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = None  # BF16 path doesn't need GradScaler

    # Train
    steps = 0
    for epoch in range(args.epochs):
        for x, y in dl:
            # --- RAG assemble ---
            if args.memory_dir and args.rag_mode != 'off':
                ms = MemoryStore(args.memory_dir)
                B, T = x.shape
                new_x = []
                new_y = []
                for i in range(B):
                    # decode a short query from the first 200 bytes
                    q = tok.decode(x[i].tolist()[:200])
                    hits = ms.search(q, top_k=args.mem_topk)
                    mem_texts = [h[0].text[:400] for h in hits]
                    mem_ids = tok.encode_memory_prefix(mem_texts)
                    # prepend or append and trim to seq_len-1 (since we shift y)
                    if args.rag_mode == 'prepend':
                        merged = (mem_ids + x[i].tolist())[:args.seq_len]
                    elif args.rag_mode == 'append':
                        merged = (x[i].tolist() + mem_ids)[:args.seq_len]
                    else:
                        merged = x[i].tolist()
                    # rebuild targets as next-token
                    merged = merged if len(merged)>=2 else merged + [tok.PAD]
                    nx = torch.tensor(merged[:-1], dtype=torch.long)
                    ny = torch.tensor(merged[1:], dtype=torch.long)
                    new_x.append(nx)
                    new_y.append(ny)
                x = torch.stack(new_x, dim=0)
                y = torch.stack(new_y, dim=0)
            # --- end RAG ---
            x, y = maybe_replay(x, y)
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, tok.vocab_size), y.view(-1))
            opt.zero_grad()
            loss.backward()
            if XLA_AVAILABLE:
                xm.optimizer_step(opt, barrier=True)
            else:
                opt.step()
            steps += 1
            if _is_master() and steps % args.log_every == 0:
                ppl = math.exp(min(loss.item(), 20))
                print(f"epoch {epoch+1} step {steps} | loss {loss.item():.4f} | ppl {ppl:.2f}")

        # save small checkpoint per epoch (master rank)
        if _is_master():
            ckpt = {
                'cfg': vars(args),
                'model_state': {k:v.detach().cpu() for k,v in model.state_dict().items()},
            }
            os.makedirs('ckpts', exist_ok=True)
            torch.save(ckpt, f'ckpts/aura_mini_e{epoch+1}.pt')
            print(f"Saved ckpt: ckpts/aura_mini_e{epoch+1}.pt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', type=str, default='demo.txt')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--seq-len', type=int, default=2048)   # 2k context for shakedown
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--attn-mode', type=str, default='streaming')
    ap.add_argument('--block-q', type=int, default=128)
    ap.add_argument('--block-kv', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--max-lines', type=int, default=None)
    ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--memory-dir', type=str, default=None)
    ap.add_argument('--mem-topk', type=int, default=3)
    ap.add_argument('--rag-mode', type=str, default='prepend', choices=['prepend', 'append', 'off'])
    ap.add_argument('--replay-ratio', type=float, default=0.0)
    ap.add_argument('--tpu', action='store_true')
    args = ap.parse_args()

    if args.tpu and XLA_AVAILABLE:
        nprocs = _world_size() or 1
        xmp.spawn(train_worker, args=(args,), nprocs=nprocs, start_method='fork')
    else:
        train_worker(0, args)

if __name__ == "__main__":
    main()
