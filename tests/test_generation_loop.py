import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp

try:
    from aura.self_teaching_llm.generation_loop import generate_text, generate_text_with_consciousness
    from aura.self_teaching_llm.spiking_retrieval_core import SpikingRetrievalCore
    from aura.self_teaching_llm.spiking_language_core import SpikingLanguageCore
    from aura.self_teaching_llm.token_decoder import TokenDecoder
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TestGenerationLoop(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures if JAX is available."""
        if not HAS_JAX:
            self.skipTest("JAX not available")
        
        # Create test key
        self.key = jax.random.key(0)
        
        # Set up component dimensions
        self.embed_dim = 32
        self.hidden_dim = 64
        self.vocab_size = 1000
        self.num_experts = 8
        
        # Create components
        self.retrieval_core = SpikingRetrievalCore(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            expert_dim=self.embed_dim
        )
        
        self.lang_core = SpikingLanguageCore(
            hidden_dim=self.hidden_dim
        )
        
        self.token_decoder = TokenDecoder(
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size
        )
        
        # Create test prompt embeddings
        self.prompt_embeddings = jax.random.normal(self.key, (2, self.embed_dim))
        
        # Initialize parameters for all components
        self.retrieval_params = self.retrieval_core.init(self.key, self.prompt_embeddings)
        self.lang_params = self.lang_core.init(self.key, self.prompt_embeddings, 
                                             self.lang_core.initialize_state(2))
        self.decoder_params = self.token_decoder.init(self.key, jnp.ones((2, self.hidden_dim)))
    
    def test_generate_text(self):
        """Test basic text generation loop."""
        if not HAS_JAX:
            return
            
        # Create mock components that work with the generation loop
        def mock_retrieval_core(query_embedding):
            # Simple linear transformation to simulate retrieval
            return jnp.dot(query_embedding, jnp.ones((self.embed_dim, self.hidden_dim)))
        
        def mock_lang_core(input_state, prev_state):
            # Simple pass-through to simulate language core
            return input_state, prev_state
        
        def mock_decoder(rate_vector):
            # Uniform distribution to simulate decoder
            return jnp.ones((rate_vector.shape[0], self.vocab_size)) / self.vocab_size
        
        # Add methods to mock objects
        mock_retrieval_core.retrieve_context = mock_retrieval_core
        mock_lang_core.initialize_state = lambda batch_size: (
            jnp.zeros((batch_size, self.hidden_dim)), 
            jnp.zeros((batch_size, self.hidden_dim))
        )
        
        # Generate text
        generated_tokens, all_rates = generate_text(
            self.prompt_embeddings,
            mock_retrieval_core,
            mock_lang_core,
            mock_decoder,
            max_len=10
        )
        
        # Check output shapes
        self.assertEqual(generated_tokens.shape, (2, 10))
        self.assertEqual(len(all_rates), 10)
        
        # Check that all rates have correct shape
        for rate in all_rates:
            self.assertEqual(rate.shape, (2, self.hidden_dim))
    
    def test_generate_text_with_consciousness(self):
        """Test text generation with consciousness integration."""
        if not HAS_JAX:
            return
            
        # Create mock components
        def mock_retrieval_core(query_embedding):
            return jnp.dot(query_embedding, jnp.ones((self.embed_dim, self.hidden_dim)))
        
        def mock_lang_core(input_state, prev_state):
            return input_state, prev_state
        
        def mock_decoder(rate_vector):
            return jnp.ones((rate_vector.shape[0], self.vocab_size)) / self.vocab_size
        
        # Add methods to mock objects
        mock_retrieval_core.retrieve_context = mock_retrieval_core
        mock_lang_core.initialize_state = lambda batch_size: (
            jnp.zeros((batch_size, self.hidden_dim)), 
            jnp.zeros((batch_size, self.hidden_dim))
        )
        
        # Create mock consciousness system
        class MockConsciousnessSystem:
            def __init__(self):
                self.workspace_manager = MockWorkspaceManager()
        
        class MockWorkspaceManager:
            def get_workspace_status(self):
                return {
                    'contents': [
                        {'id': 'content_1', 'text': 'test content', 'score': 0.8}
                    ]
                }
        
        mock_consciousness = MockConsciousnessSystem()
        
        # Generate text with consciousness
        generated_tokens, all_rates = generate_text_with_consciousness(
            self.prompt_embeddings,
            mock_retrieval_core,
            mock_lang_core,
            mock_decoder,
            consciousness_system=mock_consciousness,
            max_len=10
        )
        
        # Check output shapes
        self.assertEqual(generated_tokens.shape, (2, 10))
        self.assertEqual(len(all_rates), 10)
        
        # Check that all rates have correct shape
        for rate in all_rates:
            self.assertEqual(rate.shape, (2, self.hidden_dim))
    
    def test_temperature_scaling(self):
        """Test temperature scaling in token generation."""
        if not HAS_JAX:
            return
            
        # Create mock components
        def mock_retrieval_core(query_embedding):
            return jnp.dot(query_embedding, jnp.ones((self.embed_dim, self.hidden_dim)))
        
        def mock_lang_core(input_state, prev_state):
            return input_state, prev_state
        
        def mock_decoder(rate_vector):
            # Non-uniform distribution to test temperature effect
            logits = jnp.array([[10.0, 5.0, 1.0, 0.5] + [0.1] * (self.vocab_size - 4)])
            logits = jnp.repeat(logits, rate_vector.shape[0], axis=0)
            return jax.nn.softmax(logits, axis=-1)
        
        # Add methods to mock objects
        mock_retrieval_core.retrieve_context = mock_retrieval_core
        mock_lang_core.initialize_state = lambda batch_size: (
            jnp.zeros((batch_size, self.hidden_dim)), 
            jnp.zeros((batch_size, self.hidden_dim))
        )
        
        # Generate with high temperature (should be more uniform)
        generated_tokens_high_temp, _ = generate_text(
            self.prompt_embeddings,
            mock_retrieval_core,
            mock_lang_core,
            mock_decoder,
            max_len=5,
            temperature=2.0
        )
        
        # Generate with low temperature (should be more peaked)
        generated_tokens_low_temp, _ = generate_text(
            self.prompt_embeddings,
            mock_retrieval_core,
            mock_lang_core,
            mock_decoder,
            max_len=5,
            temperature=0.5
        )
        
        # Check output shapes
        self.assertEqual(generated_tokens_high_temp.shape, (2, 5))
        self.assertEqual(generated_tokens_low_temp.shape, (2, 5))
    
    def test_jit_compatibility(self):
        """Test that generation loop is JIT compatible."""
        if not HAS_JAX:
            return
            
        # Create mock components
        def mock_retrieval_core(query_embedding):
            return jnp.dot(query_embedding, jnp.ones((self.embed_dim, self.hidden_dim)))
        
        def mock_lang_core(input_state, prev_state):
            return input_state, prev_state
        
        def mock_decoder(rate_vector):
            return jnp.ones((rate_vector.shape[0], self.vocab_size)) / self.vocab_size
        
        # Add methods to mock objects
        mock_retrieval_core.retrieve_context = mock_retrieval_core
        mock_lang_core.initialize_state = lambda batch_size: (
            jnp.zeros((batch_size, self.hidden_dim)), 
            jnp.zeros((batch_size, self.hidden_dim))
        )
        
        # Create JIT-compiled generation function
        jit_generate = jax.jit(lambda prompt_embeddings: generate_text(
            prompt_embeddings,
            mock_retrieval_core,
            mock_lang_core,
            mock_decoder,
            max_len=10
        ))
        
        # Run JIT-compiled function
        generated_tokens, all_rates = jit_generate(self.prompt_embeddings)
        
        # Check output shapes
        self.assertEqual(generated_tokens.shape, (2, 10))
        self.assertEqual(len(all_rates), 10)


if __name__ == '__main__':
    unittest.main()
