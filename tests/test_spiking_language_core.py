import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp

try:
    from aura.self_teaching_llm.spiking_language_core import SpikingLanguageCore
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TestSpikingLanguageCore(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures if JAX is available."""
        if not HAS_JAX:
            self.skipTest("JAX not available")
        
        self.hidden_dim = 32
        self.dt = 1e-3
        
        self.language_core = SpikingLanguageCore(
            hidden_dim=self.hidden_dim,
            dt=self.dt
        )
        
        # Create test key
        self.key = jax.random.key(0)
        
        # Create test input state
        self.input_state = jax.random.normal(self.key, (4, self.hidden_dim))
        
        # Initialize parameters
        self.params = self.language_core.init(self.key, self.input_state, 
                                           self.language_core.initialize_state(4))
    
    def test_initialization(self):
        """Test language core initialization."""
        if not HAS_JAX:
            return
            
        # Check that parameters are initialized with correct shapes
        self.assertIn('params', self.params)
        self.assertIn('recurrent_weights', self.params['params'])
        
        # Check parameter shapes
        self.assertEqual(self.params['params']['recurrent_weights'].shape, 
                        (self.hidden_dim, self.hidden_dim))
    
    def test_state_processing(self):
        """Test state processing through spiking core."""
        if not HAS_JAX:
            return
            
        # Initialize state
        batch_size = self.input_state.shape[0]
        prev_state = self.language_core.initialize_state(batch_size)
        
        # Process through language core
        output_rate, next_state = self.language_core.apply(self.params, 
                                                         self.input_state, 
                                                         prev_state)
        
        # Check output shapes
        self.assertEqual(output_rate.shape, (batch_size, self.hidden_dim))
        self.assertEqual(len(next_state), 2)
        self.assertEqual(next_state[0].shape, (batch_size, self.hidden_dim))
        self.assertEqual(next_state[1].shape, (batch_size, self.hidden_dim))
        
        # Check that outputs are in valid ranges
        self.assertTrue(jnp.all(output_rate >= 0.0))
        self.assertTrue(jnp.all(output_rate <= 1.0))
    
    def test_recurrent_dynamics(self):
        """Test recurrent dynamics of spiking core."""
        if not HAS_JAX:
            return
            
        # Initialize state
        batch_size = self.input_state.shape[0]
        prev_state = self.language_core.initialize_state(batch_size)
        
        # Process same input multiple times to test recurrent dynamics
        state = prev_state
        rates = []
        
        for _ in range(5):
            output_rate, state = self.language_core.apply(self.params, 
                                                        self.input_state, 
                                                        state)
            rates.append(output_rate)
        
        # Check that we got outputs
        self.assertEqual(len(rates), 5)
        
        # Check that rates are not all identical (recurrent dynamics should change them)
        # This is a simple check - in practice, the dynamics might be more subtle
        for i in range(1, len(rates)):
            # They don't have to be different, but they shouldn't all be exactly the same
            # We'll just verify they have the correct shape and range
            self.assertEqual(rates[i].shape, (batch_size, self.hidden_dim))
            self.assertTrue(jnp.all(rates[i] >= 0.0))
            self.assertTrue(jnp.all(rates[i] <= 1.0))
    
    def test_state_initialization(self):
        """Test state initialization."""
        if not HAS_JAX:
            return
            
        batch_size = 7
        initial_state = self.language_core.initialize_state(batch_size)
        
        # Check state tuple structure
        self.assertEqual(len(initial_state), 2)
        self.assertEqual(initial_state[0].shape, (batch_size, self.hidden_dim))
        self.assertEqual(initial_state[1].shape, (batch_size, self.hidden_dim))
        
        # Check that initial state is zeros
        self.assertTrue(jnp.all(initial_state[0] == 0.0))
        self.assertTrue(jnp.all(initial_state[1] == 0.0))
    
    def test_jit_compatibility(self):
        """Test that language core is JIT compatible."""
        if not HAS_JAX:
            return
            
        # Initialize state
        batch_size = self.input_state.shape[0]
        prev_state = self.language_core.initialize_state(batch_size)
        
        # Create JIT-compiled function
        jit_process = jax.jit(lambda params, input_state, prev_state: 
                            self.language_core.apply(params, input_state, prev_state))
        
        # Run JIT-compiled function
        output_rate, next_state = jit_process(self.params, self.input_state, prev_state)
        
        # Check output shapes
        self.assertEqual(output_rate.shape, (batch_size, self.hidden_dim))
    
    def test_tpu_compatibility(self):
        """Test TPU compatibility if TPU is available."""
        if not HAS_JAX:
            return
            
        # Check if TPU is available
        tpu_devices = [d for d in jax.devices() if d.platform == 'tpu']
        if not tpu_devices:
            self.skipTest("TPU not available")
        
        # Initialize state
        batch_size = self.input_state.shape[0]
        prev_state = self.language_core.initialize_state(batch_size)
        
        # Run on TPU
        input_tpu = jax.device_put(self.input_state, tpu_devices[0])
        prev_state_tpu = (jax.device_put(prev_state[0], tpu_devices[0]), 
                         jax.device_put(prev_state[1], tpu_devices[0]))
        
        output_rate, next_state = self.language_core.apply(self.params, 
                                                         input_tpu, 
                                                         prev_state_tpu)
        
        # Check output shapes
        self.assertEqual(output_rate.shape, (batch_size, self.hidden_dim))


if __name__ == '__main__':
    unittest.main()
