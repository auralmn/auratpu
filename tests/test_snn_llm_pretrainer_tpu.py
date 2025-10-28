import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch  # noqa: E402
from snn_llm_pretrainer_tpu import SNNLLMPretrainerTPU  # noqa: E402


class TestSNNLLMPretrainerTPU(unittest.TestCase):
    def test_encode_forward_logits(self):
        trainer = SNNLLMPretrainerTPU(vocab_size=100, embedding_dim=32, hidden_neurons=16, oja_components=8, context_length=8, use_ltc=False)
        tokens = [1, 2, 3, 4]
        feat = trainer.encode_tokens(tokens)
        self.assertEqual(len(feat.shape), 2)
        spikes = trainer.forward_snn(feat, timesteps=2)
        self.assertEqual(len(spikes.shape), 3)
        logits = trainer.compute_output_logits(spikes)
        self.assertEqual(len(logits.shape), 2)


if __name__ == '__main__':
    unittest.main()
