import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import fast_clean_pack as fcp  # noqa: E402


class TestFastCleanPack(unittest.TestCase):
    def test_write_clean(self):
        tmpdir = tempfile.mkdtemp()
        try:
            inp = os.path.join(tmpdir, 'in.txt')
            outp = os.path.join(tmpdir, 'out.txt')
            with open(inp, 'w', encoding='utf-8') as w:
                w.write("Hello  world\n\nHello world\n\nUnique para")
            fcp.write_clean(inp, outp)
            with open(outp, 'r', encoding='utf-8') as r:
                data = r.read().strip()
            parts = data.split('\n\n')
            self.assertGreaterEqual(len(parts), 2)
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
