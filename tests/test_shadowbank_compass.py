import os
import sys
import io
import json
import time
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from shadowbank_compass import ShadowBank, IntentCompass  # noqa: E402


class TestShadowBankCompass(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

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

    def test_add_and_retrieve(self):
        bank = ShadowBank(self.tmp, intent_dim=64)
        t0 = time.time() - 1000.0
        bank.add_text("we solved a bug", intent_text="debugging fix", valence=0.8, tags=["insight"], ts=t0)
        bank.add_text("failed experiment", intent_text="research failure", valence=-0.5, tags=["failure"], ts=time.time())
        comp = IntentCompass(bank)
        res = comp.retrieve("fixing a bug", top_k=2)
        self.assertGreaterEqual(len(res), 1)
        # Expect the "debugging fix" to rank high
        top = res[0]
        self.assertIsNotNone(top[0].memory_id)

    def test_temporal_decay_and_valence(self):
        bank = ShadowBank(self.tmp, intent_dim=64)
        old_ts = time.time() - 3600.0
        id_old = bank.add_text("old success", intent_text="success case", valence=0.9, tags=["insight"], ts=old_ts)
        id_new = bank.add_text("new neutral", intent_text="success case", valence=0.0, tags=["note"], ts=time.time())
        comp = IntentCompass(bank)
        # Without decay, old with high valence may win with valence weighting
        r0 = comp.retrieve("success case", top_k=2, lambda_decay=0.0, use_valence=True, valence_weight=0.5)
        ids0 = [r[0].memory_id for r in r0]
        self.assertIn(id_old, ids0)
        # With decay strong, new should outrank old
        r1 = comp.retrieve("success case", top_k=2, lambda_decay=1.0, use_valence=False)
        ids1 = [r[0].memory_id for r in r1]
        self.assertEqual(ids1[0], id_new)

    def test_cli(self):
        # Add two items via CLI passthrough in main
        import shadowbank_compass as cli
        out1 = io.StringIO(); out2 = io.StringIO(); out3 = io.StringIO()
        argv1 = ['prog', 'add', '--memory-dir', self.tmp, '--text', 'alpha', '--intent', 'goal a', '--valence', '0.2']
        argv2 = ['prog', 'add', '--memory-dir', self.tmp, '--text', 'beta', '--intent', 'goal b', '--valence', '0.1']
        argv3 = ['prog', 'query', '--memory-dir', self.tmp, '--intent', 'goal a', '--top-k', '1']
        save = sys.argv
        try:
            sys.argv = argv1
            with redirect_stdout(out1):
                cli.main()
            sys.argv = argv2
            with redirect_stdout(out2):
                cli.main()
            sys.argv = argv3
            with redirect_stdout(out3):
                cli.main()
        finally:
            sys.argv = save
        res = json.loads(out3.getvalue())
        self.assertIn('results', res)
        self.assertGreaterEqual(len(res['results']), 1)


if __name__ == '__main__':
    unittest.main()
