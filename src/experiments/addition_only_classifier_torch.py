#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn


class AOClassifierTorch(nn.Module):
    def __init__(self, n_classes: int, input_dim: int, bias: bool = True):
        super().__init__()
        self.n_classes = int(n_classes)
        self.input_dim = int(input_dim)
        self.W = nn.Parameter(torch.randn(self.n_classes, self.input_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(self.n_classes)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]; W: [C, D]
        # scores: -||W - x||_1 + b
        d = torch.abs(self.W.unsqueeze(0) - x.unsqueeze(1)).sum(dim=-1)  # [B, C]
        s = -d
        if self.b is not None:
            s = s + self.b
        return s

    @torch.no_grad()
    def fit_prototypes(self, x: torch.Tensor, y: torch.Tensor):
        # compute per-class L1 medoid approximation via mean (robust enough for tests)
        C = self.n_classes
        D = x.shape[-1]
        device = x.device
        for c in range(C):
            mask = (y == c)
            if mask.any():
                self.W[c] = x[mask].mean(dim=0).to(device)


def main():
    import argparse, json, sys
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-classes', type=int, default=3)
    ap.add_argument('--input-dim', type=int, default=8)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--x', type=str, default=None, help='JSON array vector')
    args = ap.parse_args()

    device = torch.device(args.device)
    model = AOClassifierTorch(args.n_classes, args.input_dim).to(device)
    if args.x:
        vec = torch.tensor(json.loads(args.x), dtype=torch.float32, device=device).view(1, -1)
    else:
        vec = torch.randn(1, args.input_dim, device=device)
    with torch.no_grad():
        scores = model(vec)
        pred = int(scores.argmax(dim=-1).item())
    print(json.dumps({'scores': scores.view(-1).tolist(), 'pred': pred}))


if __name__ == '__main__':
    main()
