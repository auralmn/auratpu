import os
import sys
import io
import json
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from aura_mini_train_tpu import AURAMiniLM, ByteTokenizer, PreTokenizedDataset, evaluate, save_checkpoint  # noqa: E402
from neuromorphic_srwkv_tpu import get_device  # noqa: E402


class TestAuraMiniTrainTPU(unittest.TestCase):
    def test_tokenizer_and_model_forward(self):
        tok = ByteTokenizer()
        ids = tok.encode("hi", add_special=True)
        self.assertGreaterEqual(len(ids), 2)
        text = tok.decode(ids)
        self.assertIsInstance(text, str)
        V = tok.vocab_size
        device = get_device()
        model = AURAMiniLM(vocab_size=V, dim=32, heads=4, layers=2, attn_mode='streaming', block_q=8, block_kv=8).to(device)
        x = torch.randint(0, V, (2, 8), dtype=torch.long, device=device)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 8, V))

    def test_pretokenized_dataset_and_evaluate(self):
        tmpdir = tempfile.mkdtemp()
        try:
            meta = {"seq_len": 8, "vocab_size": 261, "n_samples": 2}
            with open(os.path.join(tmpdir, 'meta.json'), 'w') as w:
                json.dump(meta, w)
            T = meta['seq_len'] + 1
            toks = np.array([1,2,3,4,5,6,7,8,9] * 2, dtype=np.uint16)
            toks.tofile(os.path.join(tmpdir, 'tokens.bin'))
            idx = np.array([0, 9], dtype=np.int64)
            np.save(os.path.join(tmpdir, 'idx.npy'), idx)
            ds = PreTokenizedDataset(tmpdir)
            x, y = ds[0]
            self.assertEqual(int(x.shape[0]), meta['seq_len'])
            V = ds.vocab_size
            device = torch.device('cpu')
            model = AURAMiniLM(vocab_size=V, dim=16, heads=2, layers=1, attn_mode='streaming', block_q=4, block_kv=4).to(device)
            dl = torch.utils.data.DataLoader(ds, batch_size=1)
            ppl = evaluate(model, dl, device=device, vocab_size=V, max_steps=1)
            self.assertTrue(np.isfinite(ppl))
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

    def test_save_checkpoint(self):
        tmpdir = tempfile.mkdtemp()
        try:
            V = 64
            model = AURAMiniLM(vocab_size=V, dim=16, heads=2, layers=1, attn_mode='streaming', block_q=4, block_kv=4)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            path = os.path.join(tmpdir, 'ckpt.pt')
            save_checkpoint(path, model, opt, meta={'vocab_size': V}, log=lambda *a, **k: None)
            self.assertTrue(os.path.exists(path))
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
