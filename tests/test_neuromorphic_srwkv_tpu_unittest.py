import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch  # noqa: E402
from neuromorphic_srwkv_tpu import NeuromorphicSRWKVTpu, get_device, DEFAULT_DTYPE  # noqa: E402


class TestNeuromorphicSRWKVTpu(unittest.TestCase):
    def test_forward_and_validate(self):
        cfg = {
            'embedding_dim': 64,
            'num_heads': 4,
            'attn_mode': 'streaming',
            'block_size_q': 8,
            'block_size_kv': 8,
            'k_winners': 3,
        }
        m = NeuromorphicSRWKVTpu(cfg)
        B, T, D = 2, 16, cfg['embedding_dim']
        x = torch.randn(B, T, D, device=get_device(), dtype=DEFAULT_DTYPE)
        ids = torch.randint(0, 128, (B, T), device=get_device())
        y = m(x, ids)
        self.assertEqual(tuple(y.shape), (B, T, D))
        ok, _ = m.validate()
        self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main()
