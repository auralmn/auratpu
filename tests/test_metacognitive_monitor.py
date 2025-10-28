import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np

from aura.consciousness.metacognitive_monitor import (
    MetacognitiveMonitor,
    MetacognitiveConfig
)


class TestMetacognitiveMonitor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MetacognitiveConfig(
            history_window=10,
            calibration_window=5
        )
        self.monitor = MetacognitiveMonitor(self.config)
    
    def test_performance_and_confidence_updates(self):
        """Test updating performance and confidence tracking."""
        # Add some performance and confidence scores
        self.monitor.update_performance(0.8, 0.7)
        self.monitor.update_performance(0.9, 0.85)
        
        # Check that histories are updated
        self.assertEqual(len(self.monitor.performance_history), 2)
        self.assertEqual(len(self.monitor.confidence_scores), 2)
        
        # Check values
        self.assertIn(0.8, self.monitor.performance_history)
        self.assertIn(0.7, self.monitor.confidence_scores)
    
    def test_self_awareness_calculation(self):
        """Test self-awareness level calculation."""
        # Add enough samples to trigger calculation
        for i in range(15):
            performance = 0.7 + 0.1 * np.sin(i * 0.5)  # Varying performance
            confidence = 0.6 + 0.2 * np.cos(i * 0.3)  # Varying confidence
            self.monitor.update_performance(performance, confidence)
        
        # Self-awareness level should be calculated
        self.assertGreaterEqual(self.monitor.self_awareness_level, 0.0)
        self.assertLessEqual(self.monitor.self_awareness_level, 1.0)
    
    def test_calibration_error_calculation(self):
        """Test expected calibration error calculation."""
        # Add perfectly calibrated samples (performance = confidence)
        for i in range(10):
            score = 0.5 + 0.3 * np.sin(i * 0.5)
            self.monitor.update_performance(score, score)
        
        # Calibration error should be low
        self.assertLess(self.monitor.calibration_error, 0.1)
        
        # Clear and add poorly calibrated samples
        self.monitor.performance_history.clear()
        self.monitor.confidence_scores.clear()
        
        # High confidence but low performance
        for i in range(10):
            self.monitor.update_performance(0.2, 0.8)
        
        # Calibration error should be higher
        self.assertGreater(self.monitor.calibration_error, 0.3)
    
    def test_threshold_adjustment(self):
        """Test consciousness threshold adjustment based on self-awareness."""
        # Add samples to establish baseline self-awareness
        for i in range(15):
            performance = 0.7 + 0.1 * np.sin(i * 0.5)
            confidence = 0.6 + 0.2 * np.cos(i * 0.3)
            self.monitor.update_performance(performance, confidence)
        
        # Get adjusted thresholds
        thresholds = self.monitor.adjust_thresholds()
        
        # Check that thresholds are in expected ranges
        self.assertIn('consciousness_threshold', thresholds)
        self.assertIn('novelty_threshold', thresholds)
        
        self.assertGreaterEqual(thresholds['consciousness_threshold'], 0.5)
        self.assertLessEqual(thresholds['consciousness_threshold'], 0.8)
        
        self.assertGreaterEqual(thresholds['novelty_threshold'], 0.6)
        self.assertLessEqual(thresholds['novelty_threshold'], 0.8)
        
        # Higher self-awareness should lead to higher thresholds
        initial_awareness = self.monitor.self_awareness_level
        initial_consciousness_threshold = thresholds['consciousness_threshold']
        
        # Simulate higher self-awareness by directly setting the value
        self.monitor.self_awareness_level = 0.9
        new_thresholds = self.monitor.adjust_thresholds()
        
        # Higher self-awareness should result in higher consciousness threshold
        self.assertGreater(new_thresholds['consciousness_threshold'], 
                         initial_consciousness_threshold)
    
    def test_metacognitive_status(self):
        """Test metacognitive status reporting."""
        # Add some samples
        self.monitor.update_performance(0.8, 0.7)
        self.monitor.update_performance(0.9, 0.85)
        
        status = self.monitor.get_metacognitive_status()
        
        # Check status structure
        self.assertIn('self_awareness_level', status)
        self.assertIn('calibration_error', status)
        self.assertIn('performance_history_length', status)
        self.assertIn('confidence_history_length', status)
        self.assertIn('average_performance', status)
        self.assertIn('average_confidence', status)
        
        # Check values
        self.assertEqual(status['performance_history_length'], 2)
        self.assertEqual(status['confidence_history_length'], 2)
        self.assertGreaterEqual(status['self_awareness_level'], 0.0)
        self.assertLessEqual(status['self_awareness_level'], 1.0)
        self.assertGreaterEqual(status['calibration_error'], 0.0)
    
    def test_history_window_limit(self):
        """Test that history windows respect their limits."""
        # Add more samples than history window
        for i in range(20):
            self.monitor.update_performance(0.5, 0.5)
        
        # Should be limited to history_window size
        self.assertEqual(len(self.monitor.performance_history), self.config.history_window)
        self.assertEqual(len(self.monitor.confidence_scores), self.config.history_window)
    
    def test_empty_history_handling(self):
        """Test handling of empty histories."""
        # Status should work even with empty histories
        status = self.monitor.get_metacognitive_status()
        self.assertEqual(status['average_performance'], 0.0)
        self.assertEqual(status['average_confidence'], 0.0)
        
        # Threshold adjustment should work
        thresholds = self.monitor.adjust_thresholds()
        self.assertIn('consciousness_threshold', thresholds)
        self.assertIn('novelty_threshold', thresholds)
    
    def test_jax_compatibility(self):
        """Test that all computations are JAX-compatible."""
        # Add samples
        for i in range(10):
            performance = float(jax.random.uniform(jax.random.key(i), shape=()))
            confidence = float(jax.random.uniform(jax.random.key(i + 10), shape=()))
            self.monitor.update_performance(performance, confidence)
        
        # All operations should complete without error
        status = self.monitor.get_metacognitive_status()
        thresholds = self.monitor.adjust_thresholds()
        
        # Check that we get valid results
        self.assertIsInstance(status['self_awareness_level'], float)
        self.assertIsInstance(thresholds['consciousness_threshold'], float)


if __name__ == '__main__':
    unittest.main()
