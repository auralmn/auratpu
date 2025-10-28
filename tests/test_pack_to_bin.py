import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pack_to_bin as ptb  # noqa: E402


class TestPackToBin(unittest.TestCase):
    def test_pack(self):
        tmpdir = tempfile.mkdtemp()
        try:
            inp = os.path.join(tmpdir, 'clean.txt')
            outd = os.path.join(tmpdir, 'bin')
            with open(inp, 'w', encoding='utf-8') as w:
                w.write("A\nB\n\nC\nD\n")
            save_argv = sys.argv
            try:
                sys.argv = ['prog', '--in', inp, '--out-dir', outd, '--seq-len', '8']
                ptb.main()
            finally:
                sys.argv = save_argv
            self.assertTrue(os.path.exists(os.path.join(outd, 'tokens.bin')))
            self.assertTrue(os.path.exists(os.path.join(outd, 'idx.npy')))
            meta_path = os.path.join(outd, 'meta.json')
            self.assertTrue(os.path.exists(meta_path))
            meta = json.load(open(meta_path, 'r'))
            self.assertIn('seq_len', meta)
            self.assertIn('vocab_size', meta)
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
