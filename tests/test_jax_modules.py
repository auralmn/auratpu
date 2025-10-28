import os
import sys
import unittest
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestJaxModules(unittest.TestCase):
    def setUp(self):
        self.has_jax = importlib.util.find_spec('jax') is not None
        self.has_flax = importlib.util.find_spec('flax') is not None
        self.has_optax = importlib.util.find_spec('optax') is not None
        if not (self.has_jax and self.has_flax and self.has_optax):
            self.skipTest('JAX/Flax/Optax not installed')

    def test_neuromorphic_validate(self):
        import jax
        from neuromorphic_srwkv_jax import NeuromorphicSRWKVJax, validate_model
        model = NeuromorphicSRWKVJax(embedding_dim=32, num_heads=4, attn_mode='streaming', block_size_q=8, block_size_kv=8)
        key = jax.random.key(0)
        res = validate_model(model, key)
        self.assertTrue(res.ok, res.msg)

    def test_mini_forward(self):
        import jax
        import jax.numpy as jnp
        from aura_mini_train_jax import AURAMiniLMJax
        model = AURAMiniLMJax(vocab_size=259, dim=32, heads=4, layers=2, attn_mode='streaming', block_q=8, block_kv=8)
        key = jax.random.key(0)
        x = jnp.zeros((1, 16), dtype=jnp.int32)
        params = model.init(key, x)
        y = model.apply(params, x)
        self.assertEqual(y.shape, (1, 16, 259))


if __name__ == '__main__':
    unittest.main()
