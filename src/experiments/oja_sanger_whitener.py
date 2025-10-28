
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torch Oja/Sanger/Nonlinear Component Layer with Growth & Lateral Inhibition

- OnlineWhitener (torch): running mean/var normalization (cheap, stable)
- OjaLayer (torch):
    * mode='sanger'     : Sanger/GHA, ordered principal components
    * mode='oja'        : classic Oja (good 1st PC)
    * mode='nonlinear'  : Hyvärinen–Oja cubic Hebbian (ICA-ish)
  Extras (unchanged semantics):
    * Anti-Hebbian lateral inhibition to decorrelate co-activations
    * Residual-variance EMA growth (auto-add new component along residual)
    * Row-wise renormalization for stability
    * Save/load utilities (npz) for portability

Compat:
- Public class/method names match the original NumPy version, so existing imports work:
    from oja_sanger_whitener import OnlineWhitener, OjaLayer, NumpyEncoder
- Inputs can be NumPy arrays OR torch.Tensors.
- By default, .step() returns y as NumPy to avoid breaking existing callers.
  Set return_torch=True to get a torch.Tensor instead.

Device/dtype:
- Defaults to CPU + float32 for TPU compatibility (XLA dislikes float64).
- You can move to TPU/GPU by passing a device at construction or via .to(device, dtype).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, Any, Union

import numpy as np
import torch
import logging
import warnings

# Avoid noisy warnings during online updates
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, torch.Tensor]

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays and types (kept for compatibility)."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def _to_tensor(x: ArrayLike, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().to('cpu', torch.float64).numpy()


# ----------------------------- Whitener ------------------------------------- #
@dataclass
class WhitenerState:
    mu: np.ndarray
    var: np.ndarray
    momentum: float
    eps: float


class OnlineWhitener:
    """Featurewise online zero-mean/unit-var normalization (torch)."""

    def __init__(self, dim: int, eps: float = 1e-6, momentum: float = 0.01,
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        self._dim = int(dim)
        self._eps = float(eps)
        self._m = float(momentum)
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.mu = torch.zeros(self._dim, device=self.device, dtype=self.dtype)
        self.var = torch.ones(self._dim, device=self.device, dtype=self.dtype)
        self._min_stddev = torch.as_tensor(1e-6, device=self.device, dtype=self.dtype)

        logger.info(f"OnlineWhitener (torch) initialized: dim={dim}, device={self.device}, dtype={self.dtype}")

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def transform(self, x: ArrayLike, *, return_torch: bool = False) -> ArrayLike:
        """Update running stats and whiten a single sample x (NumPy or torch)."""
        x_t = _to_tensor(x, self.device, self.dtype)
        # Momentum update
        self.mu.mul_(1.0 - self._m).add_(x_t, alpha=self._m)
        d = x_t - self.mu
        self.var.mul_(1.0 - self._m).add_(d * d, alpha=self._m)

        # Whiten with safe stddev
        stddev = torch.sqrt(self.var)
        y = (x_t - self.mu) / (torch.maximum(stddev, self._min_stddev) + self._eps)

        if return_torch or isinstance(x, torch.Tensor):
            return y
        return _to_numpy(y)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Move whitener state to device/dtype."""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.mu = self.mu.to(self.device, self.dtype)
        self.var = self.var.to(self.device, self.dtype)
        self._min_stddev = self._min_stddev.to(self.device, self.dtype)
        return self

    def state_dict(self) -> WhitenerState:
        return WhitenerState(mu=_to_numpy(self.mu), var=_to_numpy(self.var), momentum=self._m, eps=self._eps)

    def load_state_dict(self, state: WhitenerState) -> None:
        self.mu = _to_tensor(state.mu, self.device, self.dtype).clone()
        self.var = _to_tensor(state.var, self.device, self.dtype).clone()
        self._m = float(state.momentum)
        self._eps = float(state.eps)

    def reset(self) -> None:
        self.mu.zero_()
        self.var.fill_(1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get current whitening statistics (NumPy for JSON friendliness)."""
        return {
            'dim': self._dim,
            'mu': _to_numpy(self.mu),
            'var': _to_numpy(self.var),
            'momentum': self._m,
            'eps': self._eps
        }


# ----------------------------- Oja Layer ------------------------------------ #
@dataclass
class OjaConfig:
    dim: int
    n_components: int = 8
    lr: float = 5e-4
    mode: str = "sanger"  # 'sanger' | 'oja' | 'nonlinear'
    max_components: int = 64
    lateral_beta: float = 0.05  # anti-Hebbian lateral strength
    grow_threshold: float = 0.35  # EMA residual threshold to spawn
    ema: float = 0.01  # EMA for residual
    grow_cooldown: int = 100  # steps to wait after growth
    seed: Optional[int] = None


@dataclass
class OjaStepOut:
    y: Any
    orth_error: float
    explained: float
    residual: float
    residual_ema: float
    grew: bool
    new_index: Optional[int]


class OjaLayer:
    """
    Multi-neuron Oja/Sanger/Nonlinear + growth + lateral inhibition (torch).

    Accepts NumPy or torch inputs. By default returns NumPy y for compatibility.
    """

    def __init__(self, dim: int, n_components: int = 8, lr: float = 5e-4, mode: str = "sanger",
                 *, max_components: int = 64, lateral_beta: float = 0.05,
                 grow_threshold: float = 0.35, ema: float = 0.01, grow_cooldown: int = 100,
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        self.cfg = OjaConfig(dim=dim, n_components=n_components, lr=lr, mode=mode,
                             max_components=max_components, lateral_beta=lateral_beta,
                             grow_threshold=grow_threshold, ema=ema, grow_cooldown=grow_cooldown,
                             seed=seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype

        self.dim = int(dim)
        self.lr = float(lr)
        self.mode = str(mode)
        self.max_components = int(max_components)
        self.beta = float(lateral_beta)
        self.grow_threshold = float(grow_threshold)
        self.ema = float(ema)
        self.grow_cooldown = int(grow_cooldown)
        self.cooldown = 0

        # Initialize components (rows are unit vectors)
        W0 = torch.randn(n_components, self.dim, device=self.device, dtype=self.dtype)
        self.W = (W0 / (W0.norm(dim=1, keepdim=True) + 1e-12))

        self.K = self.W.shape[0]
        self.k = self.K  # alias

        # Running stats
        self.residual_ema = 0.0
        self._steps = 0

        logger.info(f"OjaLayer (torch) initialized: mode={mode}, K={n_components}, dim={dim}, device={self.device}, dtype={self.dtype}")

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Move parameters to device/dtype."""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.W = self.W.to(self.device, self.dtype)
        return self

    # --------------------------- Core Step ---------------------------------- #
    @torch.no_grad()
    def step(self, xw: ArrayLike, *, return_torch: bool = False) -> OjaStepOut:
        """
        One online update given whitened sample xw.
        Returns OjaStepOut; y is NumPy by default (for compatibility), or torch if return_torch=True.
        """
        x = _to_tensor(xw, self.device, self.dtype)  # [dim]
        # forward
        y = self.W @ x  # (K,)
        x_hat = self.W.t() @ y  # reconstruction
        xn = float(torch.dot(x, x) + 1e-12)
        explained = float(torch.dot(x_hat, x_hat) / xn)
        residual = float(1.0 - explained)

        # learning rule
        if self.mode == "nonlinear":
            # Δw_i = α ( x*y_i^3 - w_i )
            g = y ** 3
            dW = g[:, None] * x[None, :] - self.W
        elif self.mode == "sanger":
            # Δw_i = α * y_i * ( x - sum_{j<=i} y_j w_j )
            proj = torch.zeros_like(x)
            dW = torch.zeros_like(self.W)
            for i in range(self.K):
                proj = proj + y[i] * self.W[i]
                dW[i] = y[i] * (x - proj)
        else:  # 'oja'
            # Δw_i = α * y_i * ( x - y_i * w_i )
            dW = (y[:, None] * x[None, :]) - ((y ** 2)[:, None] * self.W)

        # Anti-Hebbian lateral inhibition:
        YW = y @ self.W  # (dim,)
        cross = YW[None, :] - (y[:, None] * self.W)  # remove self-term
        dW = dW - self.beta * (y[:, None] * cross)

        # Apply update + renorm
        self.W.add_(dW, alpha=self.lr)
        self._renorm_rows_()

        # Diagnostics
        G = self.W @ self.W.t()
        orth_err = float(torch.linalg.matrix_norm(G - torch.eye(self.K, device=self.device, dtype=self.dtype)))

        # Growth logic (residual EMA)
        self.residual_ema = (1.0 - self.ema) * self.residual_ema + self.ema * residual
        grew, new_idx = self._maybe_grow_(x, x_hat)

        self._steps += 1
        y_out = y if return_torch else _to_numpy(y)
        return OjaStepOut(y=y_out, orth_error=orth_err, explained=explained, residual=residual,
                          residual_ema=self.residual_ema, grew=grew, new_index=new_idx)

    # --------------------------- Utilities ---------------------------------- #
    def components(self) -> np.ndarray:
        """Return a copy of current component matrix W (K x dim) as NumPy."""
        return _to_numpy(self.W)

    def components_tensor(self) -> torch.Tensor:
        """Torch view of current component matrix (K x dim)."""
        return self.W

    def get_components(self) -> np.ndarray:
        """Alias for components() for compatibility."""
        return self.components()

    def usage(self) -> Dict[str, float]:
        """Return simple usage diagnostics."""
        return {
            "k": float(self.K),
            "steps": float(self._steps),
            "residual_ema": float(self.residual_ema),
        }

    @torch.no_grad()
    def reset(self, keep_k: Optional[int] = None) -> None:
        """Reinitialize components (optionally keeping current K)."""
        k = keep_k if keep_k is not None else self.K
        W0 = torch.randn(k, self.dim, device=self.device, dtype=self.dtype)
        self.W = (W0 / (W0.norm(dim=1, keepdim=True) + 1e-12))
        self.K = k
        self.k = k  # Update alias
        self.residual_ema = 0.0
        self.cooldown = 0
        self._steps = 0

    # Save/load (npz) – keep NumPy portability
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        np.savez_compressed(
            path,
            W=self.components(),
            dim=np.int64(self.dim),
            K=np.int64(self.K),
            lr=np.float64(self.lr),
            mode=str(self.mode),
            max_components=np.int64(self.max_components),
            beta=np.float64(self.beta),
            grow_threshold=np.float64(self.grow_threshold),
            ema=np.float64(self.ema),
            grow_cooldown=np.int64(self.grow_cooldown),
            residual_ema=np.float64(self.residual_ema),
            steps=np.int64(self._steps),
        )

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> "OjaLayer":
        data = np.load(path, allow_pickle=True)
        layer = cls(
            dim=int(data["dim"]),
            n_components=int(data["K"]),
            lr=float(data["lr"]),
            mode=str(data["mode"]),
            max_components=int(data["max_components"]),
            lateral_beta=float(data["beta"]),
            grow_threshold=float(data["grow_threshold"]),
            ema=float(data["ema"]),
            grow_cooldown=int(data["grow_cooldown"]),
            device=device, dtype=dtype
        )
        W = torch.as_tensor(np.asarray(data["W"], dtype=np.float32), device=layer.device, dtype=layer.dtype)
        # Renorm in case of numeric drift from serialized values
        layer.W = W / (W.norm(dim=1, keepdim=True) + 1e-12)
        layer.K = layer.W.shape[0]
        layer.k = layer.K  # Update alias
        layer.residual_ema = float(data["residual_ema"])
        layer._steps = int(data["steps"])
        return layer

    # --------------------------- Internals ---------------------------------- #
    @torch.no_grad()
    def grow(self) -> int:
        """
        Manually trigger growth of a new component.
        Returns the index of the new component.
        """
        if self.K >= self.max_components:
            logger.warning(f"Cannot grow: already at max components {self.max_components}")
            return -1

        w_new = torch.randn(self.dim, device=self.device, dtype=self.dtype)
        w_new = w_new / (torch.norm(w_new) + 1e-12)
        self.W = torch.cat([self.W, w_new[None, :]], dim=0)
        self.K += 1
        self.k = self.K  # Update alias

        logger.info(f"Manually grew new component, total: {self.K}")
        return self.K - 1

    @torch.no_grad()
    def _renorm_rows_(self) -> None:
        self.W.div_(self.W.norm(dim=1, keepdim=True) + 1e-12)

    @torch.no_grad()
    def _maybe_grow_(self, x: torch.Tensor, x_hat: torch.Tensor) -> Tuple[bool, Optional[int]]:
        if self.cooldown > 0:
            self.cooldown -= 1
            return False, None
        if self.K >= self.max_components:
            return False, None
        if self.residual_ema < self.grow_threshold:
            return False, None

        # Add component along residual direction
        r = x - x_hat
        nr = torch.norm(r)
        if float(nr) > 1e-9:
            w_new = r / nr
        else:
            w_new = torch.randn_like(x)
            w_new = w_new / (torch.norm(w_new) + 1e-12)

        self.W = torch.cat([self.W, w_new[None, :]], dim=0)
        self.K += 1
        self.k = self.K  # Update alias
        self.cooldown = self.grow_cooldown
        return True, self.K - 1

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics (Python scalars)."""
        return {
            'k': self.K,
            'dim': self.dim,
            'mode': self.mode,
            'learning_rate': self.lr,
            'max_components': self.max_components,
            'lateral_beta': self.beta,
            'grow_threshold': self.grow_threshold,
            'ema': self.ema,
            'grow_cooldown': self.grow_cooldown,
            'residual_ema': float(self.residual_ema),
            'steps': self._steps,
            'cooldown': self.cooldown
        }
