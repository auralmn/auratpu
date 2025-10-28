import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp

from aura.consciousness.memory_processor import AURAMemoryProcessor, MemoryProcessorConfig


class TestAURAMemoryProcessor(unittest.TestCase):
    
    def test_fragment_knowledge(self):
        """Test knowledge fragmentation into overlapping chunks."""
        config = MemoryProcessorConfig(fragment_size=3, max_fragments=1000, perturbation_std=0.01)
        processor = AURAMemoryProcessor(config)
        
        text = "the quick brown fox jumps over the lazy dog"
        embedding = jnp.ones(10)  # Simple embedding for testing
        
        fragments = processor.fragment_knowledge(text, embedding)
        
        # Should have correct number of fragments (9 words, fragment_size=3)
        expected_count = len(text.split()) - config.fragment_size + 1
        self.assertEqual(len(fragments), expected_count)
        
        # Check fragment structure
        for fragment in fragments:
            self.assertIn('id', fragment)
            self.assertIn('text', fragment)
            self.assertIn('embedding', fragment)
            self.assertIn('strength', fragment)
            self.assertIn('timestamp', fragment)
            
            # Text should have correct number of words
            self.assertEqual(len(fragment['text'].split()), config.fragment_size)
            
            # Embedding should be perturbed but same shape
            self.assertEqual(fragment['embedding'].shape, embedding.shape)
    
    def test_short_text_handling(self):
        """Test handling of text shorter than fragment size."""
        config = MemoryProcessorConfig(fragment_size=10, max_fragments=1000, perturbation_std=0.01)
        processor = AURAMemoryProcessor(config)
        
        text = "short text"
        embedding = jnp.ones(5)
        
        fragments = processor.fragment_knowledge(text, embedding)
        
        # Should return empty list for text shorter than fragment size
        self.assertEqual(len(fragments), 0)
    
    def test_embedding_perturbation(self):
        """Test that embeddings are perturbed correctly."""
        config = MemoryProcessorConfig(fragment_size=3, perturbation_std=0.1)
        processor = AURAMemoryProcessor(config)
        
        original_embedding = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        perturbed_embedding = processor._apply_semantic_perturbation(original_embedding)
        
        # Should be same shape
        self.assertEqual(perturbed_embedding.shape, original_embedding.shape)
        
        # Should be different (perturbed) but close to original
        diff = jnp.abs(perturbed_embedding - original_embedding).max()
        self.assertGreater(diff, 0)  # Should have some perturbation
        self.assertLess(diff, 1.0)   # Should be relatively small perturbation


if __name__ == '__main__':
    unittest.main()
