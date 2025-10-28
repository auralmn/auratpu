#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


class AOClassifierJax(nn.Module):
    n_classes: int
    input_dim: int
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, D]
        W = self.param('W', nn.initializers.normal(stddev=0.1), (self.n_classes, self.input_dim))
        if self.bias:
            b = self.param('b', nn.initializers.zeros, (self.n_classes,))
        else:
            b = None
        d = jnp.sum(jnp.abs(W[None, :, :] - x[:, None, :]), axis=-1)  # [B, C]
        s = -d
        if b is not None:
            s = s + b
        return s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-classes', type=int, default=3)
    ap.add_argument('--input-dim', type=int, default=8)
    ap.add_argument('--x', type=str, default=None, help='JSON array vector')
    args = ap.parse_args()

    model = AOClassifierJax(args.n_classes, args.input_dim)
    key = jax.random.key(0)
    if args.x:
        vec = jnp.array(json.loads(args.x), dtype=jnp.float32)[None, :]
    else:
        vec = jax.random.normal(key, (1, args.input_dim), dtype=jnp.float32)
    params = model.init(key, vec)
    scores = model.apply(params, vec)
    pred = int(jnp.argmax(scores, axis=-1)[0])
    print(json.dumps({'scores': list(map(float, scores[0])), 'pred': pred}))


if __name__ == '__main__':
    main()
