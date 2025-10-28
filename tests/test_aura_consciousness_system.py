import os
import sys
import unittest
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp

from aura.consciousness.aura_consciousness_system import (
    AURAConsciousnessSystem,
    ConsciousnessConfig,
    ProcessingState
)


class TestAURAConsciousnessSystem(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ConsciousnessConfig(
            theta_frequency=10.0,  # Faster for testing
            gamma_frequency=50.0,  # Faster for testing
            workspace_capacity=3,
            consciousness_threshold=0.1,  # Lower for testing
            novelty_threshold=0.1  # Lower for testing
        )
        self.consciousness_system = AURAConsciousnessSystem(self.config)
    
    def test_initialization(self):
        """Test consciousness system initialization."""
        # Check that all components are initialized
        self.assertIsNotNone(self.consciousness_system.oscillator)
        self.assertIsNotNone(self.consciousness_system.memory_processor)
        self.assertIsNotNone(self.consciousness_system.workspace_manager)
        self.assertIsNotNone(self.consciousness_system.dream_synthesizer)
        self.assertIsNotNone(self.consciousness_system.metacognitive_monitor)
        
        # Check initial state
        self.assertEqual(self.consciousness_system.processing_state, ProcessingState.WAKE)
        self.assertFalse(self.consciousness_system.is_processing)
    
    def test_processing_state_transitions(self):
        """Test processing state transitions."""
        # Test setting different states
        self.consciousness_system.set_processing_state(ProcessingState.REM_SLEEP)
        self.assertEqual(self.consciousness_system.processing_state, ProcessingState.REM_SLEEP)
        
        self.consciousness_system.set_processing_state(ProcessingState.DREAM_SYNTHESIS)
        self.assertEqual(self.consciousness_system.processing_state, ProcessingState.DREAM_SYNTHESIS)
        
        self.consciousness_system.set_processing_state(ProcessingState.DEEP_SLEEP)
        self.assertEqual(self.consciousness_system.processing_state, ProcessingState.DEEP_SLEEP)
    
    def test_add_knowledge(self):
        """Test adding knowledge to the system."""
        # Create test embedding
        embedding = jnp.array([0.5, 0.3, 0.2, 0.1, 0.0])
        
        # Add knowledge
        self.consciousness_system.add_knowledge("Test knowledge", embedding)
        
        # Check that knowledge was added to buffer
        self.assertEqual(len(self.consciousness_system.knowledge_buffer), 1)
        
        # Check knowledge item structure
        item = self.consciousness_system.knowledge_buffer[0]
        self.assertIn('text', item)
        self.assertIn('embedding', item)
        self.assertIn('timestamp', item)
        self.assertEqual(item['text'], "Test knowledge")
        self.assertEqual(item['embedding'].shape, embedding.shape)
    
    def test_consciousness_status(self):
        """Test consciousness status reporting."""
        status = self.consciousness_system.get_consciousness_status()
        
        # Check status structure
        self.assertIn('processing_state', status)
        self.assertIn('is_processing', status)
        self.assertIn('consciousness_level', status)
        self.assertIn('calibration_error', status)
        self.assertIn('knowledge_buffer_size', status)
        self.assertIn('workspace', status)
        self.assertIn('adjusted_thresholds', status)
        self.assertIn('oscillator', status)
        
        # Check initial values
        self.assertEqual(status['processing_state'], 'wake')
        self.assertFalse(status['is_processing'])
        self.assertEqual(status['knowledge_buffer_size'], 0)
    
    def test_workspace_clearing_in_deep_sleep(self):
        """Test that workspace is cleared when entering deep sleep."""
        # Add some content to workspace
        from aura.consciousness.global_workspace_manager import ConsciousContent
        content = ConsciousContent(
            id="test_content",
            text="Test content",
            embedding=jnp.ones(10),
            timestamp=time.time(),
            relevance=0.8,
            novelty=0.6,
            emotional_salience=0.7,
            strength=1.0
        )
        
        self.consciousness_system.workspace_manager.compete_for_consciousness([content])
        self.assertEqual(len(self.consciousness_system.workspace_manager.workspace), 1)
        
        # Set to deep sleep and check workspace is cleared
        self.consciousness_system.set_processing_state(ProcessingState.DEEP_SLEEP)
        self.assertEqual(len(self.consciousness_system.workspace_manager.workspace), 0)
    
    def test_processing_loop_starts_and_stops(self):
        """Test that processing loop can be started and stopped."""
        # Start processing
        self.consciousness_system.start_processing()
        
        # Check that processing started
        self.assertTrue(self.consciousness_system.is_processing)
        self.assertIsNotNone(self.consciousness_system.processing_thread)
        
        # Add some knowledge to ensure processing occurs
        embedding = jnp.array([0.5, 0.3, 0.2, 0.1, 0.0])
        self.consciousness_system.add_knowledge("Test knowledge", embedding)
        
        # Wait a moment for processing to occur
        time.sleep(0.1)
        
        # Stop processing
        self.consciousness_system.stop_processing()
        
        # Check that processing stopped
        self.assertFalse(self.consciousness_system.is_processing)
    
    def test_oscillator_integration(self):
        """Test oscillator integration and phase updates."""
        # Start processing
        self.consciousness_system.start_processing()
        
        # Wait for oscillator phases to update
        time.sleep(0.1)
        
        # Get status and check oscillator info
        status = self.consciousness_system.get_consciousness_status()
        self.assertIn('oscillator', status)
        
        oscillator_status = status['oscillator']
        self.assertIn('theta_phase', oscillator_status)
        self.assertIn('gamma_phase', oscillator_status)
        self.assertIn('in_coupling_window', oscillator_status)
        
        # Stop processing
        self.consciousness_system.stop_processing()
    
    def test_metacognitive_integration(self):
        """Test metacognitive monitor integration."""
        # Add knowledge to trigger metacognitive updates
        embedding = jnp.array([0.5, 0.3, 0.2, 0.1, 0.0])
        self.consciousness_system.add_knowledge("Test knowledge", embedding)
        
        # Start processing to trigger metacognitive updates
        self.consciousness_system.start_processing()
        time.sleep(0.1)
        self.consciousness_system.stop_processing()
        
        # Check metacognitive status
        status = self.consciousness_system.get_consciousness_status()
        self.assertIn('consciousness_level', status)
        self.assertIn('calibration_error', status)
        
        # Values should be in valid ranges
        self.assertGreaterEqual(status['consciousness_level'], 0.0)
        self.assertLessEqual(status['consciousness_level'], 1.0)
        self.assertGreaterEqual(status['calibration_error'], 0.0)
        self.assertLessEqual(status['calibration_error'], 1.0)
    
    def test_threshold_adjustment_integration(self):
        """Test threshold adjustment integration."""
        # Get status and check adjusted thresholds
        status = self.consciousness_system.get_consciousness_status()
        self.assertIn('adjusted_thresholds', status)
        
        thresholds = status['adjusted_thresholds']
        self.assertIn('consciousness_threshold', thresholds)
        self.assertIn('novelty_threshold', thresholds)
        
        # Thresholds should be in expected ranges
        self.assertGreaterEqual(thresholds['consciousness_threshold'], 0.5)
        self.assertLessEqual(thresholds['consciousness_threshold'], 0.8)
        self.assertGreaterEqual(thresholds['novelty_threshold'], 0.6)
        self.assertLessEqual(thresholds['novelty_threshold'], 0.8)


if __name__ == '__main__':
    unittest.main()
