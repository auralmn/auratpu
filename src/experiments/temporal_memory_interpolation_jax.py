#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


def analytic_signal(x: jnp.ndarray) -> jnp.ndarray:
    """Return analytic signal using frequency-domain Hilbert transform.
    x: [..., T] real
    Returns: complex analytic signal z with same shape.
    """
    X = jnp.fft.fft(x)
    N = x.shape[-1]
    h = jnp.zeros(N, dtype=X.dtype)
    if N % 2 == 0:
        # even
        h = h.at[0].set(1.0 + 0j)
        h = h.at[N // 2].set(1.0 + 0j)
        h = h.at[1:N // 2].set(2.0 + 0j)
    else:
        # odd
        h = h.at[0].set(1.0 + 0j)
        h = h.at[1:(N + 1) // 2].set(2.0 + 0j)
    Z = jnp.fft.ifft(X * h)
    return Z


def hamiltonian_evolve(z: jnp.ndarray, omega: float, t: float) -> jnp.ndarray:
    """Phase rotation for harmonic Hamiltonian evolution."""
    phase = jnp.exp(1j * jnp.asarray(omega * t, dtype=z.dtype))
    return z * phase


def fourier_resample(z: jnp.ndarray, up: int) -> jnp.ndarray:
    """Simple Fourier-domain zero-padding resampling by integer upsample factor.
    z: [..., T] complex
    """
    T = z.shape[-1]
    Z = jnp.fft.fft(z)
    # Zero-pad in frequency domain
    pad = (up - 1) * T
    Zp = jnp.concatenate([Z[..., : T // 2], jnp.zeros(Z.shape[:-1] + (pad + (T % 2 == 0),), Z.dtype), Z[..., T // 2 + (T % 2 == 0):]], axis=-1)
    zp = jnp.fft.ifft(Zp) * up
    return zp


def interpolate_temporal_memory(x: jnp.ndarray, omega: float, dt: float, up: int = 1) -> jnp.ndarray:
    """Pipeline: analytic signal -> Hamiltonian evolve by dt -> optional Fourier resample -> real part.
    x: [..., T] real
    Returns array with last dim T*up.
    """
    z = analytic_signal(x)
    z_e = hamiltonian_evolve(z, omega, dt)
    if up != 1:
        z_e = fourier_resample(z_e, up=up)
    return jnp.real(z_e)


@dataclass
class TmiResult:
    x_out: jnp.ndarray
    energy_preserved: float
    phase_shift: float


def tmi_demo(T: int = 128, omega: float = 0.3, dt: float = 1.0, up: int = 1) -> TmiResult:
    key = jax.random.key(0)
    x = jax.random.normal(key, (T,), dtype=jnp.float32)
    z = analytic_signal(x)
    z_e = hamiltonian_evolve(z, omega, dt)
    if up != 1:
        z_e = fourier_resample(z_e, up)
    x2 = jnp.real(z_e)
    energy = float(jnp.sqrt(jnp.mean(jnp.square(x))) / (jnp.sqrt(jnp.mean(jnp.square(x2))) + 1e-12))
    return TmiResult(x_out=x2, energy_preserved=energy, phase_shift=omega * dt)
