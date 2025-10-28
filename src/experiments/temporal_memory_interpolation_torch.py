#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def analytic_signal(x: torch.Tensor) -> torch.Tensor:
    """Analytic signal via frequency-domain Hilbert transform.
    x: [..., T] real tensor
    returns complex tensor of same shape
    """
    X = torch.fft.fft(x)
    N = x.shape[-1]
    h = torch.zeros(N, dtype=X.dtype, device=X.device)
    if N % 2 == 0:
        h[0] = 1.0 + 0j
        h[N // 2] = 1.0 + 0j
        h[1:N // 2] = 2.0 + 0j
    else:
        h[0] = 1.0 + 0j
        h[1:(N + 1) // 2] = 2.0 + 0j
    Z = torch.fft.ifft(X * h)
    return Z


def hamiltonian_evolve(z: torch.Tensor, omega: float, t: float) -> torch.Tensor:
    phase = torch.exp(1j * torch.as_tensor(omega * t, dtype=z.dtype, device=z.device))
    return z * phase


def fourier_resample(z: torch.Tensor, up: int) -> torch.Tensor:
    T = z.shape[-1]
    Z = torch.fft.fft(z)
    pad = (up - 1) * T
    even = (T % 2 == 0)
    left = Z[..., : T // 2]
    right = Z[..., T // 2 + (1 if even else 0):]
    midpad = torch.zeros(z.shape[:-1] + (pad + (1 if even else 0),), dtype=Z.dtype, device=Z.device)
    Zp = torch.cat([left, midpad, right], dim=-1)
    zp = torch.fft.ifft(Zp) * up
    return zp


def interpolate_temporal_memory(x: torch.Tensor, omega: float, dt: float, up: int = 1) -> torch.Tensor:
    z = analytic_signal(x)
    z_e = hamiltonian_evolve(z, omega, dt)
    if up != 1:
        z_e = fourier_resample(z_e, up)
    return torch.real(z_e)


def tmi_demo(device: torch.device = torch.device('cpu'), T: int = 128, omega: float = 0.3, dt: float = 1.0, up: int = 1):
    x = torch.randn(T, device=device, dtype=torch.float32)
    x2 = interpolate_temporal_memory(x, omega, dt, up)
    energy_ratio = (x.pow(2).mean().sqrt() / (x2.pow(2).mean().sqrt() + 1e-12)).item()
    return x2, energy_ratio
