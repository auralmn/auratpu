import os
import sys
import unittest
import importlib.util
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestAOClassifierTorch(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
            self.torch = importlib.import_module('torch')
            self.has_torch = True
        except Exception:
            self.has_torch = False
            self.skipTest('torch not installed')

    def test_forward_cpu(self):
        from addition_only_classifier_torch import AOClassifierTorch
        device = self.torch.device('cpu')
        model = AOClassifierTorch(n_classes=3, input_dim=8).to(device)
        x = self.torch.randn(4, 8, device=device)
        y = model(x)
        self.assertEqual(tuple(y.shape), (4, 3))

    def test_forward_mps_if_available(self):
        if not self.torch.backends.mps.is_available():
            self.skipTest('MPS not available')
        from addition_only_classifier_torch import AOClassifierTorch
        device = self.torch.device('mps')
        model = AOClassifierTorch(n_classes=3, input_dim=8).to(device)
        x = self.torch.randn(4, 8, device=device)
        y = model(x)
        self.assertEqual(tuple(y.shape), (4, 3))

    def test_fit_prototypes_improves(self):
        from addition_only_classifier_torch import AOClassifierTorch
        torch = self.torch
        device = torch.device('cpu')
        rng = torch.Generator().manual_seed(0)
        # Two clusters around centers c0, c1
        c0 = torch.zeros(8)
        c1 = torch.ones(8)
        X0 = c0 + 0.05 * torch.randn(50, 8, generator=rng)
        X1 = c1 + 0.05 * torch.randn(50, 8, generator=rng)
        X = torch.cat([X0, X1], dim=0).to(device)
        y = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)], dim=0).to(device)
        model = AOClassifierTorch(n_classes=2, input_dim=8).to(device)
        with torch.no_grad():
            base_pred = model(X).argmax(dim=-1)
            base_acc = (base_pred == y).float().mean().item()
        model.fit_prototypes(X, y)
        with torch.no_grad():
            pred = model(X).argmax(dim=-1)
            acc = (pred == y).float().mean().item()
        self.assertGreaterEqual(acc, base_acc)


class TestAOClassifierJax(unittest.TestCase):
    def setUp(self):
        self.has_jax = importlib.util.find_spec('jax') is not None
        self.has_flax = importlib.util.find_spec('flax') is not None
        if not (self.has_jax and self.has_flax):
            self.skipTest('JAX/Flax not installed')

    def test_forward_cpu_or_gpu(self):
        import jax
        import jax.numpy as jnp
        from addition_only_classifier_jax import AOClassifierJax
        model = AOClassifierJax(n_classes=3, input_dim=8)
        key = jax.random.key(0)
        x = jax.random.normal(key, (4, 8), dtype=jnp.float32)
        params = model.init(key, x)
        y = model.apply(params, x)
        self.assertEqual(y.shape, (4, 3))

    def test_tpu_if_available(self):
        import jax
        if not any(d.platform == 'tpu' for d in jax.devices()):
            self.skipTest('TPU not available for JAX')
        import jax.numpy as jnp
        from addition_only_classifier_jax import AOClassifierJax
        model = AOClassifierJax(n_classes=2, input_dim=8)
        key = jax.random.key(0)
        x = jax.random.normal(key, (2, 8), dtype=jnp.float32)
        params = model.init(key, x)
        y = model.apply(params, x)
        self.assertEqual(y.shape, (2, 2))

    def test_jit_compat(self):
        import jax
        import jax.numpy as jnp
        from addition_only_classifier_jax import AOClassifierJax
        model = AOClassifierJax(n_classes=3, input_dim=8)
        key = jax.random.key(1)
        x = jnp.zeros((1, 8), dtype=jnp.float32)
        params = model.init(key, x)
        fn = jax.jit(lambda v: model.apply(params, v))
        y = fn(x)
        self.assertEqual(y.shape, (1, 3))


class TestAOClassifierMaxTextIntegration(unittest.TestCase):
    def setUp(self):
        self.has_jax = importlib.util.find_spec('jax') is not None
        self.has_maxtext = importlib.util.find_spec('maxtext') is not None
        if not (self.has_jax and self.has_maxtext):
            self.skipTest('MaxText not installed')

    def test_basic_jit(self):
        import jax
        import jax.numpy as jnp
        from addition_only_classifier_jax import AOClassifierJax
        model = AOClassifierJax(n_classes=3, input_dim=8)
        key = jax.random.key(0)
        x = jnp.zeros((2, 8), dtype=jnp.float32)
        params = model.init(key, x)
        fn = jax.jit(lambda v: model.apply(params, v))
        y = fn(x)
        self.assertEqual(y.shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
