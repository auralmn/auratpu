#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


def get_device():
    devs = jax.devices()
    return devs[0] if devs else None

DEFAULT_DTYPE = jnp.bfloat16 if any(d.platform == 'tpu' for d in jax.devices()) else jnp.float32


def _split_heads(x: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    B, T, D = x.shape
    Hd = D // num_heads
    return x.reshape(B, T, num_heads, Hd).transpose(0, 2, 1, 3)


def _merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    B, H, T, Hd = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * Hd)


class NeuromorphicSRWKVJax(nn.Module):
    embedding_dim: int
    num_heads: int
    attn_mode: str = 'streaming'  # 'streaming' | 'chunked' | 'dot'
    block_size_q: int = 64
    block_size_kv: int = 64
    spike_threshold: float = 0.5
    decay_factor: float = 0.9

    @nn.compact
    def __call__(self, x: jnp.ndarray, token_ids: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        D = self.embedding_dim
        H = self.num_heads
        q = nn.Dense(D, use_bias=False, name='q')(x)
        k = nn.Dense(D, use_bias=False, name='k')(x)
        v = nn.Dense(D, use_bias=False, name='v')(x)

        qh = _split_heads(q, H)
        kh = _split_heads(k, H)
        vh = _split_heads(v, H)

        if self.attn_mode == 'streaming':
            out_h = self.streaming_attention(qh, kh, vh, self.block_size_q, self.block_size_kv)
        elif self.attn_mode == 'chunked':
            out_h = self.chunked_attention(qh, kh, vh, self.block_size_q, self.block_size_kv)
        else:
            out_h = self.scaled_dot_attention(qh, kh, vh)

        out = _merge_heads(out_h)
        out = nn.Dense(D, name='o')(out)
        return out

    def scaled_dot_attention(self, qh: jnp.ndarray, kh: jnp.ndarray, vh: jnp.ndarray) -> jnp.ndarray:
        Dh = qh.shape[-1]
        scale = 1.0 / jnp.sqrt(Dh)
        scores = jnp.einsum('bhtd,bhkd->bhtk', qh, kh) * scale
        attn = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum('bhtk,bhkd->bhtd', attn, vh)

    def chunked_attention(self, qh: jnp.ndarray, kh: jnp.ndarray, vh: jnp.ndarray, bq: int, bkv: int) -> jnp.ndarray:
        B, H, T, Dh = qh.shape
        outputs = []
        for qs in range(0, T, bq):
            qe = min(qs + bq, T)
            q_blk = qh[:, :, qs:qe, :]
            scores_acc = []
            for ks in range(0, T, bkv):
                ke = min(ks + bkv, T)
                k_blk = kh[:, :, ks:ke, :]
                s = jnp.einsum('bhtd,bhkd->bhtk', q_blk, k_blk)
                scores_acc.append(s)
            scores = jnp.concatenate(scores_acc, axis=-1)
            attn = jax.nn.softmax(scores, axis=-1)
            out_blk = jnp.zeros((B, H, qe - qs, Dh), dtype=attn.dtype)
            cursor = 0
            for ks in range(0, T, bkv):
                ke = min(ks + bkv, T)
                v_blk = vh[:, :, ks:ke, :]
                a_slice = attn[:, :, :, cursor:cursor + (ke - ks)]
                out_blk = out_blk + jnp.einsum('bhtk,bhkd->bhtd', a_slice, v_blk)
                cursor += (ke - ks)
            outputs.append(out_blk)
        return jnp.concatenate(outputs, axis=2)

    def streaming_attention(self, qh: jnp.ndarray, kh: jnp.ndarray, vh: jnp.ndarray, bq: int, bkv: int) -> jnp.ndarray:
        B, H, T, Dh = qh.shape
        scale = 1.0 / jnp.sqrt(Dh)
        outs = []
        for qs in range(0, T, bq):
            qe = min(qs + bq, T)
            q_blk = qh[:, :, qs:qe, :].astype(jnp.float32)
            m = jnp.full((B, H, qe - qs, 1), -jnp.inf, dtype=jnp.float32)
            Z = jnp.zeros((B, H, qe - qs, 1), dtype=jnp.float32)
            O = jnp.zeros((B, H, qe - qs, Dh), dtype=jnp.float32)
            for ks in range(0, T, bkv):
                ke = min(ks + bkv, T)
                k_blk = kh[:, :, ks:ke, :].astype(jnp.float32)
                v_blk = vh[:, :, ks:ke, :].astype(jnp.float32)
                scores = jnp.einsum('bhtd,bhkd->bhtk', q_blk, k_blk) * scale
                m_block = jnp.max(scores, axis=-1, keepdims=True)
                m_new = jnp.maximum(m, m_block)
                alpha = jnp.exp(m - m_new)
                exp_scores = jnp.exp(scores - m_new)
                Z = Z * alpha + jnp.sum(exp_scores, axis=-1, keepdims=True)
                O = O * alpha + jnp.einsum('bhtk,bhkd->bhtd', exp_scores, v_blk)
                m = m_new
            out_blk = (O / (Z + 1e-9)).astype(qh.dtype)
            outs.append(out_blk)
        return jnp.concatenate(outs, axis=2)


@dataclass
class ValidateResult:
    ok: bool
    msg: str


def validate_model(model: NeuromorphicSRWKVJax, key: jax.random.KeyArray) -> ValidateResult:
    B, T, D = 2, 8, model.embedding_dim
    x = jax.random.normal(key, (B, T, D), dtype=DEFAULT_DTYPE)
    ids = jax.random.randint(key, (B, T), 0, 1024)
    variables = model.init(key, x, ids)
    y = model.apply(variables, x, ids)
    if y.shape != x.shape:
        return ValidateResult(False, f"shape {y.shape} != {x.shape}")
    if not jnp.isfinite(y.astype(jnp.float32)).all():
        return ValidateResult(False, "NaN/Inf in output")
    return ValidateResult(True, "OK")
