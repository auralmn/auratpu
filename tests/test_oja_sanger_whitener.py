import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oja_sanger_whitener import OnlineWhitener, OjaLayer  # noqa: E402


class TestWhitenerAndOja(unittest.TestCase):
    def test_whitener_transform(self):
        ow = OnlineWhitener(dim=4)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = ow.transform(x)
        self.assertEqual(y.shape, (4,))

    def test_oja_step_and_grow(self):
        layer = OjaLayer(dim=4, n_components=2, lr=1e-2, mode='sanger')
        x = np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32)
        out = layer.step(x)
        self.assertIn('y', out.__dict__)
        k_before = layer.K
        layer.grow()
        self.assertEqual(layer.K, k_before + 1)


if __name__ == '__main__':
    unittest.main()
