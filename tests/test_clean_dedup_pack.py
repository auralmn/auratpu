import os
import sys
import tempfile
import unittest
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import clean_dedup_pack as cdp  # noqa: E402


class TestCleanDedupPack(unittest.TestCase):
    def test_process_optional_mmh3(self):
        if importlib.util.find_spec('mmh3') is None:
            self.skipTest('mmh3 not available')
        tmpdir = tempfile.mkdtemp()
        try:
            inp = os.path.join(tmpdir, 'in.txt')
            outp = os.path.join(tmpdir, 'out.txt')
            with open(inp, 'w', encoding='utf-8') as w:
                w.write("Para A\n\nPara B\n\nPara A")
            cdp.process(inp, outp)
            self.assertTrue(os.path.exists(outp))
            with open(outp, 'r', encoding='utf-8') as r:
                data = r.read().strip()
            self.assertGreater(len(data), 0)
        finally:
            try:
                for root, dirs, files in os.walk(tmpdir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(tmpdir)
            except Exception:
                pass


if __name__ == '__main__':
    unittest.main()
