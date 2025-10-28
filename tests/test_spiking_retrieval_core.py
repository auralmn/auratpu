import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp

try:
    from aura.self_teaching_llm.spiking_retrieval_core import SpikingRetrievalCore
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TestSpikingRetrievalCore(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures if JAX is available."""
        if not HAS_JAX:
            self.skipTest("JAX not available")
        
        self.hidden_dim = 64
        self.num_experts = 8
        self.expert_dim = 32
        
        self.retrieval_core = SpikingRetrievalCore(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            expert_dim=self.expert_dim
        )
        
        # Create test key
        self.key = jax.random.key(0)
        
        # Create test query embedding
        self.query_embedding = jax.random.normal(self.key, (4, self.expert_dim))
        
        # Initialize parameters
        self.params = self.retrieval_core.init(self.key, self.query_embedding)
    
    def test_initialization(self):
        """Test retrieval core initialization."""
        if not HAS_JAX:
            return
            
        # Check that experts are initialized with correct shape
        self.assertIn('params', self.params)
        self.assertIn('experts', self.params['params'])
        self.assertIn('gate', self.params['params'])
        
        # Check expert shape: [num_experts, expert_dim, hidden_dim]
        experts_shape = self.params['params']['experts'].shape
        self.assertEqual(experts_shape, (self.num_experts, self.expert_dim, self.hidden_dim))
        
        # Check gate shape: [expert_dim, num_experts] (transposed)
        gate_shape = self.params['params']['gate']['kernel'].shape
        self.assertEqual(gate_shape, (self.expert_dim, self.num_experts))
    
    def test_context_retrieval(self):
        """Test context retrieval from neuromorphic memory."""
        if not HAS_JAX:
            return
            
        # Retrieve context
        context_vector = self.retrieval_core.apply(self.params, self.query_embedding)
        
        # Check output shape
        self.assertEqual(context_vector.shape, (4, self.hidden_dim))
        
        # Check that output is not all zeros
        self.assertGreater(jnp.sum(jnp.abs(context_vector)), 0)
    
    def test_retrieve_context_method(self):
        """Test the retrieve_context method."""
        if not HAS_JAX:
            return
            
        # Retrieve context using method
        context_vector = self.retrieval_core.apply(self.params, 
                                                 self.query_embedding, 
                                                 method=self.retrieval_core.retrieve_context)
        
        # Check output shape
        self.assertEqual(context_vector.shape, (4, self.hidden_dim))
        
        # Check that output is not all zeros
        self.assertGreater(jnp.sum(jnp.abs(context_vector)), 0)
    
    def test_gate_probability_distribution(self):
        """Test that gate probabilities sum to 1."""
        if not HAS_JAX:
            return
            
        # Get gating logits and probabilities
        gate_logits = self.retrieval_core.apply(self.params, 
                                              self.query_embedding, 
                                              method=lambda module, x: module.gate(x))
        gate_probs = jax.nn.softmax(gate_logits)
        
        # Check that probabilities sum to 1 for each batch item
        prob_sums = jnp.sum(gate_probs, axis=1)
        expected_sums = jnp.ones_like(prob_sums)
        # Use jax.numpy.allclose instead of jnp.testing.assert_allclose
        self.assertTrue(jnp.allclose(prob_sums, expected_sums, rtol=1e-6))
    
    def test_expert_output_shapes(self):
        """Test that expert outputs have correct shapes."""
        if not HAS_JAX:
            return
            
        # Get intermediate outputs
        def get_expert_outputs(module, x):
            gate_logits = module.gate(x)
            gate_probs = jax.nn.softmax(gate_logits)
            
            # Apply expert transformations
            expert_outputs = []
            for i in range(module.num_experts):
                # Apply each expert transformation
                expert_output = jnp.dot(x, module.experts[i])  # [batch, hidden_dim]
                expert_outputs.append(expert_output)
            
            # Stack expert outputs
            expert_outputs = jnp.stack(expert_outputs, axis=1)  # [batch, num_experts, hidden_dim]
            
            return expert_outputs, gate_probs
        
        expert_outputs, gate_probs = self.retrieval_core.apply(self.params, 
                                                             self.query_embedding, 
                                                             method=get_expert_outputs)
        
        # Check shapes
        self.assertEqual(expert_outputs.shape, (4, self.num_experts, self.hidden_dim))
        self.assertEqual(gate_probs.shape, (4, self.num_experts))
    
    def test_jit_compatibility(self):
        """Test that retrieval core is JIT compatible."""
        if not HAS_JAX:
            return
            
        # Create JIT-compiled function
        jit_retrieve = jax.jit(lambda params, query: self.retrieval_core.apply(params, query))
        
        # Run JIT-compiled function
        context_vector = jit_retrieve(self.params, self.query_embedding)
        
        # Check output shape
        self.assertEqual(context_vector.shape, (4, self.hidden_dim))
    
    def test_tpu_compatibility(self):
        """Test TPU compatibility if TPU is available."""
        if not HAS_JAX:
            return
            
        # Check if TPU is available
        tpu_devices = [d for d in jax.devices() if d.platform == 'tpu']
        if not tpu_devices:
            self.skipTest("TPU not available")
        
        # Run on TPU
        context_vector = jax.device_put(self.query_embedding, tpu_devices[0])
        context_vector = self.retrieval_core.apply(self.params, context_vector)
        
        # Check output shape
        self.assertEqual(context_vector.shape, (4, self.hidden_dim))


if __name__ == '__main__':
    unittest.main()
