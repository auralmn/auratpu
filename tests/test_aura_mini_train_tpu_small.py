import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch  # noqa: E402
from aura_mini_train_tpu_small import AURAMiniLM  # noqa: E402
from neuromorphic_srwkv_tpu import get_device  # noqa: E402


class TestAuraMiniSmall(unittest.TestCase):
    def test_forward(self):
        device = get_device()
        model = AURAMiniLM(vocab_size=259, dim=16, heads=2, layers=1, attn_mode='streaming', block_q=4, block_kv=4).to(device)
        x = torch.randint(0, 259, (2, 8), dtype=torch.long, device=device)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 8, 259))


if __name__ == '__main__':
    unittest.main()
