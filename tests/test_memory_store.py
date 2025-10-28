import os
import sys
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory_store import MemoryStore  # noqa: E402


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        try:
            for root, dirs, files in os.walk(self.tmpdir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.tmpdir)
        except Exception:
            pass

    def test_add_and_search(self):
        ms = MemoryStore(self.tmpdir)
        hid = ms.add_text("hello world", title="t")
        self.assertTrue(hid)
        res = ms.search("hello", top_k=1)
        self.assertGreaterEqual(len(res), 1)

    def test_ingest_url_patched(self):
        ms = MemoryStore(self.tmpdir)
        try:
            import memory_store as ms_mod  # noqa
        except Exception as e:
            self.skipTest(f"cannot import memory_store: {e}")
        import unittest.mock as mock
        with mock.patch('memory_store._fetch_url', return_value=("Title", "This is some long text " * 20)):
            hid = ms.ingest_url('http://example.com')
            self.assertTrue(hid)


class TestAuraMemoryIngestCLI(unittest.TestCase):
    def test_cli_single_url(self):
        import unittest.mock as mock
        import aura_memory_ingest as ami
        tmpdir = tempfile.mkdtemp()
        try:
            buf = io.StringIO()
            with mock.patch('memory_store._fetch_url', return_value=("Title", "hello world " * 20)):
                argv = ['prog', '--memory-dir', tmpdir, '--url', 'http://example.com']
                with redirect_stdout(buf):
                    save_argv = sys.argv
                    try:
                        sys.argv = argv
                        ami.main()
                    finally:
                        sys.argv = save_argv
            out = json.loads(buf.getvalue())
            self.assertIn('total_items', out)
            self.assertGreaterEqual(out['total_items'], 1)
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
