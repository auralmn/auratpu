import os
import sys
import unittest
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np

from aura.consciousness.dream_synthesizer import (
    AURADreamSynthesizer, 
    DreamSynthesizerConfig,
    ConsciousContent
)


class TestAURADreamSynthesizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DreamSynthesizerConfig(
            novelty_threshold=0.1,  # Lower threshold for testing
            max_synthesis_attempts=5,
            synthesis_fragment_count=3
        )
        self.synthesizer = AURADreamSynthesizer(self.config)
        
        # Create test fragments with overlapping embeddings to allow synthesis
        self.test_fragments = [
            {
                'id': 'frag_1',
                'text': 'the quick brown',
                'embedding': jnp.array([0.6, 0.3, 0.2, 0.1, 0.0]),
                'timestamp': time.time() - 100,
                'strength': 1.0
            },
            {
                'id': 'frag_2',
                'text': 'fox jumps over',
                'embedding': jnp.array([0.3, 0.6, 0.2, 0.1, 0.0]),
                'timestamp': time.time() - 50,
                'strength': 0.8
            },
            {
                'id': 'frag_3',
                'text': 'the lazy dog',
                'embedding': jnp.array([0.2, 0.2, 0.7, 0.1, 0.0]),
                'timestamp': time.time(),
                'strength': 0.9
            },
            {
                'id': 'frag_4',
                'text': 'machine learning',
                'embedding': jnp.array([0.1, 0.1, 0.2, 0.8, 0.2]),
                'timestamp': time.time() - 25,
                'strength': 1.0
            },
            {
                'id': 'frag_5',
                'text': 'artificial intelligence',
                'embedding': jnp.array([0.0, 0.1, 0.1, 0.2, 0.8]),
                'timestamp': time.time() - 75,
                'strength': 0.7
            }
        ]
    
    def test_synthesize_concepts(self):
        """Test concept synthesis with diverse fragments."""
        gamma_amplitude = 0.8
        synthesized = self.synthesizer.synthesize_concepts(self.test_fragments, gamma_amplitude)
        
        # Should produce some synthesized contents
        self.assertGreater(len(synthesized), 0)
        
        # Check structure of synthesized contents
        for content in synthesized:
            self.assertIsInstance(content, ConsciousContent)
            self.assertIsNotNone(content.id)
            self.assertIsNotNone(content.text)
            self.assertIsNotNone(content.embedding)
            self.assertGreater(content.novelty, self.config.novelty_threshold)
    
    def test_select_diverse_fragments(self):
        """Test fragment selection for maximum diversity."""
        selected = self.synthesizer._select_diverse_fragments(self.test_fragments, 3)
        
        # Should select 3 fragments
        self.assertEqual(len(selected), 3)
        
        # Should select different fragments (orthogonal embeddings)
        fragment_ids = [fragment['id'] for fragment in selected]
        self.assertEqual(len(set(fragment_ids)), 3)  # All unique
    
    def test_combine_fragments(self):
        """Test fragment combination with weighting."""
        gamma_amplitude = 0.5
        synthesis_vector = self.synthesizer._combine_fragments(self.test_fragments[:3], gamma_amplitude)
        
        # Should have same dimension as input embeddings
        self.assertEqual(synthesis_vector.shape[0], self.test_fragments[0]['embedding'].shape[0])
        
        # Should not be all zeros (combination occurred)
        self.assertGreater(jnp.sum(jnp.abs(synthesis_vector)), 0)
    
    def test_calculate_novelty(self):
        """Test novelty calculation based on similarity."""
        # Create a synthesis vector that's very different from fragments
        synthesis_vector = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])  # Different from fragment embeddings
        novelty_score = self.synthesizer._calculate_novelty(synthesis_vector, self.test_fragments[:2])
        
        # Should produce a novelty score
        self.assertIsInstance(novelty_score, float)
        self.assertGreaterEqual(novelty_score, 0.0)
        self.assertLessEqual(novelty_score, 1.0)
    
    def test_calculate_similarity(self):
        """Test cosine similarity calculation."""
        # Test identical vectors (similarity = 1)
        vec1 = jnp.array([1.0, 0.0, 0.0])
        vec2 = jnp.array([1.0, 0.0, 0.0])
        similarity = self.synthesizer._calculate_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Test orthogonal vectors (similarity = 0)
        vec3 = jnp.array([1.0, 0.0, 0.0])
        vec4 = jnp.array([0.0, 1.0, 0.0])
        similarity = self.synthesizer._calculate_similarity(vec3, vec4)
        self.assertAlmostEqual(similarity, 0.0, places=5)
        
        # Test opposite vectors (similarity = 0 due to clipping)
        vec5 = jnp.array([1.0, 0.0, 0.0])
        vec6 = jnp.array([-1.0, 0.0, 0.0])
        similarity = self.synthesizer._calculate_similarity(vec5, vec6)
        self.assertAlmostEqual(similarity, 0.0, places=5)  # Clipped to 0
    
    def test_create_conscious_content(self):
        """Test creation of conscious content from fragments."""
        synthesis_vector = jnp.array([0.5, 0.5, 0.0, 0.0, 0.0])
        novelty_score = 0.8
        
        content = self.synthesizer._create_conscious_content(
            self.test_fragments[:2], 
            synthesis_vector, 
            novelty_score
        )
        
        # Check content structure
        self.assertIsInstance(content, ConsciousContent)
        self.assertIsNotNone(content.id)
        self.assertIsNotNone(content.text)
        self.assertEqual(content.embedding.shape, synthesis_vector.shape)
        self.assertEqual(content.novelty, novelty_score)
        
        # Check that text combines fragment texts
        self.assertIn(self.test_fragments[0]['text'], content.text)
        self.assertIn(self.test_fragments[1]['text'], content.text)
    
    def test_novelty_threshold_filtering(self):
        """Test that only sufficiently novel contents are synthesized."""
        # Create fragments with very similar embeddings
        similar_fragments = [
            {
                'id': 'frag_a',
                'text': 'similar content one',
                'embedding': jnp.array([1.0, 0.0, 0.0]),
                'timestamp': time.time(),
                'strength': 1.0
            },
            {
                'id': 'frag_b',
                'text': 'similar content two',
                'embedding': jnp.array([0.9, 0.1, 0.0]),
                'timestamp': time.time(),
                'strength': 1.0
            }
        ]
        
        # Set high novelty threshold
        high_threshold_config = DreamSynthesizerConfig(novelty_threshold=0.9)
        high_threshold_synthesizer = AURADreamSynthesizer(high_threshold_config)
        
        # Try to synthesize with very similar fragments
        synthesized = high_threshold_synthesizer.synthesize_concepts(similar_fragments, 1.0)
        
        # Should produce fewer or no contents due to high threshold
        self.assertLess(len(synthesized), len(similar_fragments))
    
    def test_empirical_diversity_validation(self):
        """Empirical validation that diverse fragment selection works correctly."""
        # Set random seed for reproducible test
        random.seed(42)
        
        # Create fragments with known similarity structure
        diverse_fragments = [
            {
                'id': 'orthogonal_1',
                'text': 'orthogonal one',
                'embedding': jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                'timestamp': time.time(),
                'strength': 1.0
            },
            {
                'id': 'orthogonal_2',
                'text': 'orthogonal two',
                'embedding': jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]),
                'timestamp': time.time(),
                'strength': 1.0
            },
            {
                'id': 'orthogonal_3',
                'text': 'orthogonal three',
                'embedding': jnp.array([0.0, 0.0, 1.0, 0.0, 0.0]),
                'timestamp': time.time(),
                'strength': 1.0
            },
            {
                'id': 'similar_to_1',
                'text': 'similar to one',
                'embedding': jnp.array([0.9, 0.0, 0.1, 0.0, 0.0]),
                'timestamp': time.time(),
                'strength': 1.0
            }
        ]
        
        # Select diverse fragments
        selected = self.synthesizer._select_diverse_fragments(diverse_fragments, 3)
        
        # Should prefer orthogonal fragments over similar ones
        selected_ids = [fragment['id'] for fragment in selected]
        
        # Check that we selected the orthogonal set (1, 2, 3) rather than including similar_to_1
        orthogonal_selected = (
            'orthogonal_1' in selected_ids and 
            'orthogonal_2' in selected_ids and 
            'orthogonal_3' in selected_ids
        )
        
        # This test validates the empirical diversity selection mechanism
        self.assertTrue(orthogonal_selected, 
                       "Diverse fragment selection should prefer orthogonal embeddings")
    
    def test_empirical_synthesis_quality(self):
        """Empirical validation of synthesis quality through statistical measures."""
        # Run multiple synthesis attempts to gather statistics
        all_novelty_scores = []
        synthesis_count = 10
        
        for _ in range(synthesis_count):
            # Set random seed for reproducibility in this test
            random.seed(42)
            
            synthesized = self.synthesizer.synthesize_concepts(self.test_fragments, 0.7)
            for content in synthesized:
                all_novelty_scores.append(content.novelty)
        
        # Convert to numpy array for statistical analysis
        novelty_scores = np.array(all_novelty_scores)
        
        # Validate statistical properties
        if len(novelty_scores) > 0:
            # Mean novelty should be above threshold
            mean_novelty = np.mean(novelty_scores)
            self.assertGreaterEqual(mean_novelty, self.config.novelty_threshold,
                                  f"Mean novelty score {mean_novelty} should be above threshold {self.config.novelty_threshold}")
            
            # Novelty scores should be in valid range [0, 1]
            self.assertTrue(np.all(novelty_scores >= 0.0), 
                          "All novelty scores should be >= 0.0")
            self.assertTrue(np.all(novelty_scores <= 1.0), 
                          "All novelty scores should be <= 1.0")


if __name__ == '__main__':
    unittest.main()
