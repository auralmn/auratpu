import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp

try:
    from aura.self_teaching_llm.self_teaching_adapter import SelfTeachingAdapter
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TestSelfTeachingAdapter(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures if JAX is available."""
        if not HAS_JAX:
            self.skipTest("JAX not available")
        
        # Create adapter with small dimensions for testing
        self.adapter = SelfTeachingAdapter(
            embed_dim=32,
            hidden_dim=64,
            vocab_size=1000,
            num_experts=8
        )
    
    def test_initialization(self):
        """Test self-teaching adapter initialization."""
        if not HAS_JAX:
            return
            
        # Check that components are initialized
        self.assertIsNotNone(self.adapter.retrieval_core)
        self.assertIsNotNone(self.adapter.lang_core)
        self.assertIsNotNone(self.adapter.token_decoder)
        
        # Check dimensions
        self.assertEqual(self.adapter.embed_dim, 32)
        self.assertEqual(self.adapter.hidden_dim, 64)
        self.assertEqual(self.adapter.vocab_size, 1000)
        self.assertEqual(self.adapter.num_experts, 8)
        
        # Check that parameters are not yet initialized
        self.assertIsNone(self.adapter.retrieval_params)
        self.assertIsNone(self.adapter.lang_params)
        self.assertIsNone(self.adapter.decoder_params)
    
    def test_parameter_initialization(self):
        """Test parameter initialization for all components."""
        if not HAS_JAX:
            return
            
        # Initialize parameters
        self.adapter.initialize_parameters(batch_size=2)
        
        # Check that parameters are now initialized
        self.assertIsNotNone(self.adapter.retrieval_params)
        self.assertIsNotNone(self.adapter.lang_params)
        self.assertIsNotNone(self.adapter.decoder_params)
        
        # Check that parameters have expected structure
        self.assertIn('params', self.adapter.retrieval_params)
        self.assertIn('params', self.adapter.lang_params)
        self.assertIn('params', self.adapter.decoder_params)
    
    def test_generate_with_consciousness(self):
        """Test text generation with consciousness integration."""
        if not HAS_JAX:
            return
            
        # Create test prompt embeddings
        prompt_embeddings = jax.random.normal(jax.random.key(0), (2, 32))
        
        # Generate text (without consciousness system)
        generated_tokens, all_rates = self.adapter.generate_with_consciousness(
            prompt_embeddings,
            max_len=10
        )
        
        # Check output shapes
        self.assertEqual(generated_tokens.shape, (2, 10))
        self.assertEqual(len(all_rates), 10)
        
        # Check that all rates have correct shape
        for rate in all_rates:
            self.assertEqual(rate.shape, (2, 64))
    
    def test_component_status(self):
        """Test component status reporting."""
        if not HAS_JAX:
            return
            
        # Get initial status
        status = self.adapter.get_component_status()
        
        # Check status structure
        self.assertIn('embed_dim', status)
        self.assertIn('hidden_dim', status)
        self.assertIn('vocab_size', status)
        self.assertIn('num_experts', status)
        self.assertIn('parameters_initialized', status)
        
        # Check initial values
        self.assertEqual(status['embed_dim'], 32)
        self.assertEqual(status['hidden_dim'], 64)
        self.assertEqual(status['vocab_size'], 1000)
        self.assertEqual(status['num_experts'], 8)
        self.assertFalse(status['parameters_initialized'])
        
        # Initialize parameters and check status again
        self.adapter.initialize_parameters(batch_size=2)
        status = self.adapter.get_component_status()
        self.assertTrue(status['parameters_initialized'])
    
    def test_jit_compatibility(self):
        """Test that adapter is JIT compatible."""
        if not HAS_JAX:
            return
            
        # Initialize parameters
        self.adapter.initialize_parameters(batch_size=2)
        
        # Create test prompt embeddings
        prompt_embeddings = jax.random.normal(jax.random.key(0), (2, 32))
        
        # Create JIT-compiled generation function
        jit_generate = jax.jit(lambda prompt: self.adapter.generate_with_consciousness(
            prompt, max_len=10))
        
        # Run JIT-compiled function
        generated_tokens, all_rates = jit_generate(prompt_embeddings)
        
        # Check output shapes
        self.assertEqual(generated_tokens.shape, (2, 10))
        self.assertEqual(len(all_rates), 10)


if __name__ == '__main__':
    unittest.main()
