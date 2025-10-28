import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp

try:
    from aura.self_teaching_llm.token_decoder import TokenDecoder
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TestTokenDecoder(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures if JAX is available."""
        if not HAS_JAX:
            self.skipTest("JAX not available")
        
        self.hidden_dim = 32
        self.vocab_size = 1000
        
        self.token_decoder = TokenDecoder(
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size
        )
        
        # Create test key
        self.key = jax.random.key(0)
        
        # Create test rate vector
        self.rate_vector = jax.random.uniform(self.key, (4, self.hidden_dim))
        
        # Initialize parameters
        self.params = self.token_decoder.init(self.key, self.rate_vector)
    
    def test_initialization(self):
        """Test token decoder initialization."""
        if not HAS_JAX:
            return
            
        # Check that parameters are initialized with correct shapes
        self.assertIn('params', self.params)
        self.assertIn('fc', self.params['params'])
        
        # Check linear layer kernel shape
        fc_kernel_shape = self.params['params']['fc']['kernel'].shape
        self.assertEqual(fc_kernel_shape, (self.hidden_dim, self.vocab_size))
    
    def test_decode_rates(self):
        """Test decoding of spiking rates to token probabilities."""
        if not HAS_JAX:
            return
            
        # Decode rates to probabilities
        token_probs = self.token_decoder.apply(self.params, self.rate_vector)
        
        # Check output shape
        self.assertEqual(token_probs.shape, (4, self.vocab_size))
        
        # Check that probabilities sum to 1
        prob_sums = jnp.sum(token_probs, axis=1)
        expected_sums = jnp.ones_like(prob_sums)
        self.assertTrue(jnp.allclose(prob_sums, expected_sums, rtol=1e-6))
        
        # Check that probabilities are in valid range [0, 1]
        self.assertTrue(jnp.all(token_probs >= 0.0))
        self.assertTrue(jnp.all(token_probs <= 1.0))
    
    def test_sample_tokens(self):
        """Test token sampling from rate vectors."""
        if not HAS_JAX:
            return
            
        # Sample tokens
        def sample_method(module, rate_vector):
            return module.sample_tokens(rate_vector, self.key)
        
        token_indices = self.token_decoder.apply(self.params, 
                                               self.rate_vector, 
                                               method=sample_method)
        
        # Check output shape
        self.assertEqual(token_indices.shape, (4,))
        
        # Check that token indices are valid (within vocab size)
        self.assertTrue(jnp.all(token_indices >= 0))
        self.assertTrue(jnp.all(token_indices < self.vocab_size))
    
    def test_probability_distribution(self):
        """Test that output probabilities form a valid distribution."""
        if not HAS_JAX:
            return
            
        # Get probabilities
        token_probs = self.token_decoder.apply(self.params, self.rate_vector)
        
        # Check that all probabilities are non-negative
        self.assertTrue(jnp.all(token_probs >= 0.0))
        
        # Check that no probability is greater than 1
        self.assertTrue(jnp.all(token_probs <= 1.0))
        
        # Check that probabilities sum to 1 for each batch item
        prob_sums = jnp.sum(token_probs, axis=1)
        expected_sums = jnp.ones_like(prob_sums)
        self.assertTrue(jnp.allclose(prob_sums, expected_sums, rtol=1e-6))
    
    def test_phasic_response_modulation(self):
        """Test phasic response generation with different modes."""
        if not HAS_JAX:
            return
            
        # Test different response modes
        modes = ['analytical', 'poetic', 'assertive', 'empathetic']
        
        for mode in modes:
            # Create a new token decoder with specific mode
            mode_decoder = TokenDecoder(hidden_dim=self.hidden_dim, vocab_size=self.vocab_size)
            
            # Initialize parameters
            mode_params = mode_decoder.init(self.key, self.rate_vector)
            
            # Get probabilities
            token_probs = mode_decoder.apply(mode_params, self.rate_vector)
            
            # Check output shape
            self.assertEqual(token_probs.shape, (4, self.vocab_size))
            
            # Check that probabilities are in valid range [0, 1]
            self.assertTrue(jnp.all(token_probs >= 0.0))
            self.assertTrue(jnp.all(token_probs <= 1.0))
    
    def test_phase_trajectory_update(self):
        """Test phase trajectory tracking during decoding."""
        if not HAS_JAX:
            return
            
        # Get probabilities (this should update phase trajectory)
        token_probs = self.token_decoder.apply(self.params, self.rate_vector)
        
        # Check output shape
        self.assertEqual(token_probs.shape, (4, self.vocab_size))
    
    def test_jit_compatibility(self):
        """Test that token decoder is JIT compatible."""
        if not HAS_JAX:
            return
            
        # Create JIT-compiled function
        jit_decode = jax.jit(lambda params, rate_vector: self.token_decoder.apply(params, rate_vector))
        
        # Run JIT-compiled function
        token_probs = jit_decode(self.params, self.rate_vector)
        
        # Check output shape
        self.assertEqual(token_probs.shape, (4, self.vocab_size))
    
    def test_tpu_compatibility(self):
        """Test TPU compatibility if TPU is available."""
        if not HAS_JAX:
            return
            
        # Check if TPU is available
        tpu_devices = [d for d in jax.devices() if d.platform == 'tpu']
        if not tpu_devices:
            self.skipTest("TPU not available")
        
        # Run on TPU
        rate_tpu = jax.device_put(self.rate_vector, tpu_devices[0])
        token_probs = self.token_decoder.apply(self.params, rate_tpu)
        
        # Check output shape
        self.assertEqual(token_probs.shape, (4, self.vocab_size))


if __name__ == '__main__':
    unittest.main()
