import os
import sys
import json
import tempfile
import unittest
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _make_pt(tmpdir: str):
    import torch
    d_model = 4
    event_names = ["event_a", "event_b", "event_c"]
    keyword_list = ["alpha", "beta", "gamma"]
    keyword_indices = {"alpha": 0, "beta": 1, "gamma": 2}
    # Simple patterns and scoring
    event_patterns = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float32)
    scoring_matrix = torch.tensor([
        [2.0, 0.0, 0.0],   # alpha contributes to event_a
        [0.0, 3.0, 0.0],   # beta -> event_b
        [0.0, 0.0, 1.0],   # gamma -> event_c
    ], dtype=torch.float32)  # [num_keywords, num_events]
    data = {
        'metadata': {'d_model': d_model},
        'event_names': event_names,
        'keyword_list': keyword_list,
        'keyword_indices': keyword_indices,
        'event_patterns': event_patterns,
        'scoring_matrix': scoring_matrix,
        'temporal_patterns': {},
    }
    pt_path = os.path.join(tmpdir, 'enc.pt')
    torch.save(data, pt_path)
    return pt_path, data


class TestFastEventEncoderTorch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        try:
            import torch  # noqa: F401
            self.has_torch = True
        except Exception:
            self.has_torch = False
            self.skipTest('torch not installed')

    def tearDown(self):
        try:
            for root, dirs, files in os.walk(self.tmp, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.tmp)
        except Exception:
            pass

    def test_encode_and_analysis(self):
        from fast_event_encoder import FastEventPatternEncoder
        pt_path, meta = _make_pt(self.tmp)
        enc = FastEventPatternEncoder(pt_file_path=pt_path)
        pat = enc.encode_text_to_patterns("alpha gamma")
        self.assertEqual(tuple(pat.shape), (1, meta['metadata']['d_model']))
        a = enc.get_event_analysis("alpha beta beta")
        self.assertIn('detected_events', a)
        self.assertGreaterEqual(a['total_keywords_found'], 1)


class TestFastEventEncoderJax(unittest.TestCase):
    def setUp(self):
        self.has_jax = importlib.util.find_spec('jax') is not None
        try:
            import torch  # noqa
            self.has_torch = True
        except Exception:
            self.has_torch = False
        if not (self.has_jax and self.has_torch):
            self.skipTest('JAX and torch required')

    def test_parity_with_torch(self):
        import jax.numpy as jnp
        from fast_event_encoder import FastEventPatternEncoder
        from fast_event_encoder_jax import FastEventPatternEncoderJax
        pt_path, meta = _make_pt(tempfile.mkdtemp())
        try:
            enc_t = FastEventPatternEncoder(pt_file_path=pt_path)
            enc_j = FastEventPatternEncoderJax(pt_file_path=pt_path)
            text = "alpha alpha gamma"
            pt = enc_t.encode_text_to_patterns(text).detach().cpu().numpy()
            pj = jnp.array(enc_j.encode_text_to_patterns(text))
            # Shapes equal and values close
            self.assertEqual(tuple(pt.shape), tuple(pj.shape))
            diff = abs(pt - pj).max()
            self.assertLessEqual(float(diff), 1e-5)
        finally:
            try:
                os.remove(pt_path)
            except Exception:
                pass


if __name__ == '__main__':
    unittest.main()
