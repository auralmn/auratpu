import os
import sys
import unittest
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestTmiTorch(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
            self.has_torch = True
        except Exception:
            self.has_torch = False
            self.skipTest('torch not installed')

    def test_cpu(self):
        import torch
        from temporal_memory_interpolation_torch import tmi_demo
        x2, er = tmi_demo(device=torch.device('cpu'), T=64, omega=0.2, dt=1.0, up=1)
        self.assertEqual(x2.shape[-1], 64)
        self.assertTrue(float(er) > 0)

    def test_mps_if_available(self):
        import torch
        if not torch.backends.mps.is_available():
            self.skipTest('MPS not available')
        from temporal_memory_interpolation_torch import tmi_demo
        try:
            x2, er = tmi_demo(device=torch.device('mps'), T=64, omega=0.2, dt=1.0, up=1)
            self.assertEqual(x2.shape[-1], 64)
            self.assertTrue(float(er) > 0)
        except NotImplementedError as e:
            self.skipTest(f'MPS FFT not supported: {e}')


class TestTmiJax(unittest.TestCase):
    def setUp(self):
        self.has_jax = importlib.util.find_spec('jax') is not None
        if not self.has_jax:
            self.skipTest('jax not installed')

    def test_jax_cpu_or_gpu(self):
        import jax
        from temporal_memory_interpolation_jax import tmi_demo
        res = tmi_demo(T=64, omega=0.2, dt=1.0, up=1)
        self.assertEqual(int(res.x_out.shape[-1]), 64)
        self.assertTrue(float(res.energy_preserved) > 0)

    def test_jax_tpu_if_available(self):
        import jax
        if not any(d.platform == 'tpu' for d in jax.devices()):
            self.skipTest('TPU not available for JAX')
        from temporal_memory_interpolation_jax import tmi_demo
        res = tmi_demo(T=64, omega=0.2, dt=1.0, up=1)
        self.assertEqual(int(res.x_out.shape[-1]), 64)
        self.assertTrue(float(res.energy_preserved) > 0)


class TestTmiMaxTextIntegration(unittest.TestCase):
    def setUp(self):
        # Optional integration: only run if MaxText is importable
        self.has_jax = importlib.util.find_spec('jax') is not None
        self.has_maxtext = importlib.util.find_spec('maxtext') is not None
        if not (self.has_jax and self.has_maxtext):
            self.skipTest('MaxText not installed')

    def test_jit_compat(self):
        import jax
        import jax.numpy as jnp
        from temporal_memory_interpolation_jax import interpolate_temporal_memory
        x = jnp.zeros((1, 64), dtype=jnp.float32)
        fn = jax.jit(lambda arr: interpolate_temporal_memory(arr, omega=0.2, dt=1.0, up=1))
        y = fn(x)
        self.assertEqual(y.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
