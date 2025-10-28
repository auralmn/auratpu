#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import List, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

from neuromorphic_srwkv_jax import NeuromorphicSRWKVJax, DEFAULT_DTYPE


class ByteTokenizer:
    def __init__(self):
        self.PAD, self.BOS, self.EOS = 256, 257, 258
        self.vocab_size = 259
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids = list(text.encode('utf-8', errors='ignore'))
        if add_special:
            return [self.BOS] + ids + [self.EOS]
        return ids


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-5
    @nn.compact
    def __call__(self, x):
        w = self.param('weight', nn.initializers.ones, (self.dim,))
        ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        norm = jax.lax.rsqrt(ms + self.eps)
        return w * x * norm


class SwiGLU(nn.Module):
    dim: int
    hidden_mult: float = 4.0
    @nn.compact
    def __call__(self, x):
        h = int(self.dim * self.hidden_mult)
        w1 = nn.Dense(h, use_bias=False)(x)
        w2 = nn.Dense(h, use_bias=False)(x)
        y = nn.Dense(self.dim, use_bias=False)(jax.nn.silu(w1) * w2)
        return y


class AURABlockJax(nn.Module):
    dim: int
    heads: int
    attn_mode: str
    block_q: int
    block_kv: int
    @nn.compact
    def __call__(self, x, token_ids: Optional[jnp.ndarray] = None):
        cfg = dict(
            embedding_dim=self.dim,
            num_heads=self.heads,
            attn_mode=self.attn_mode,
            block_size_q=self.block_q,
            block_size_kv=self.block_kv,
        )
        n1 = RMSNorm(self.dim)(x)
        mix = NeuromorphicSRWKVJax(**cfg)(n1, token_ids)
        x = x + mix
        x = x + SwiGLU(self.dim, hidden_mult=4.0)(RMSNorm(self.dim)(x))
        return x


class AURAMiniLMJax(nn.Module):
    vocab_size: int
    dim: int
    heads: int
    layers: int
    attn_mode: str
    block_q: int
    block_kv: int
    @nn.compact
    def __call__(self, token_ids: jnp.ndarray):
        x = nn.Embed(self.vocab_size, self.dim)(token_ids)
        for _ in range(self.layers):
            x = AURABlockJax(self.dim, self.heads, self.attn_mode, self.block_q, self.block_kv)(x, token_ids)
        x = RMSNorm(self.dim)(x)
        logits = nn.Dense(self.vocab_size, use_bias=False)(x)
        return logits


def loss_fn(params, model, batch):
    x, y = batch
    logits = model.apply(params, x)
    V = logits.shape[-1]
    loss = optax.softmax_cross_entropy(
        logits.reshape(-1, V), jax.nn.one_hot(y.reshape(-1,), V)
    ).mean()
    return loss


def make_toy_data(tok: ByteTokenizer, seq_len=16, batches=2):
    import numpy as np
    ids = tok.encode("hello world", add_special=True) * ((seq_len + 1) // 4)
    ids = (ids + [tok.PAD])[:seq_len+1]
    x = np.array(ids[:-1], dtype='int32')[None, :]
    y = np.array(ids[1:], dtype='int32')[None, :]
    return [(x, y) for _ in range(batches)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-len', type=int, default=16)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--attn-mode', type=str, default='streaming')
    ap.add_argument('--block-q', type=int, default=8)
    ap.add_argument('--block-kv', type=int, default=8)
    ap.add_argument('--steps', type=int, default=3)
    args = ap.parse_args()

    tok = ByteTokenizer()
    model = AURAMiniLMJax(
        vocab_size=tok.vocab_size,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        attn_mode=args.attn_mode,
        block_q=args.block_q,
        block_kv=args.block_kv,
    )

    key = jax.random.key(0)
    x0 = jnp.zeros((1, args.seq_len), dtype=jnp.int32)
    params = model.init(key, x0)

    opt = optax.adamw(2e-4)
    state = opt.init(params)

    @jax.jit
    def step(params, state, batch):
        l, grads = jax.value_and_grad(loss_fn)(params, model, batch)
        updates, state2 = opt.update(grads, state)
        params2 = optax.apply_updates(params, updates)
        return params2, state2, l

    data = make_toy_data(tok, seq_len=args.seq_len, batches=args.steps)
    for i, (x_np, y_np) in enumerate(data, 1):
        x = jnp.array(x_np)
        y = jnp.array(y_np)
        params, state, l = step(params, state, (x, y))
        print(f"step {i} loss {float(l):.4f}")


if __name__ == '__main__':
    main()
