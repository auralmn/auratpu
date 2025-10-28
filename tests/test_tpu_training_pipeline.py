#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the TPU training pipeline
"""

import os
import sys
import unittest
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from aura.training.tpu_training_pipeline import (
        TrainingConfig, AURATrainingPipeline
    )
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TestTPUTrainingPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        if not HAS_JAX:
            self.skipTest("JAX not available")
        
        self.config = TrainingConfig()
        # Use temporary directory for checkpoints
        self.config.local_checkpoint_dir = tempfile.mkdtemp()
    
    def test_training_config_initialization(self):
        """Test that training configuration initializes correctly"""
        self.assertEqual(self.config.phase0_epochs, 10)
        self.assertEqual(self.config.phase1_epochs, 20)
        self.assertEqual(self.config.phase2_epochs, 30)
        self.assertEqual(self.config.embed_dim, 768)
        self.assertEqual(self.config.hidden_dim, 512)
        self.assertEqual(self.config.vocab_size, 32000)
        self.assertEqual(self.config.num_experts, 16)
    
    def test_pipeline_initialization(self):
        """Test that training pipeline initializes correctly"""
        pipeline = AURATrainingPipeline(self.config)
        
        # Check that components are initialized
        self.assertIsNotNone(pipeline.consciousness)
        self.assertIsNotNone(pipeline.adapter)
        
        # Check that metrics structure exists
        self.assertIn('phase0', pipeline.metrics)
        self.assertIn('phase1', pipeline.metrics)
        self.assertIn('phase2', pipeline.metrics)
    
    def test_dummy_data_creation(self):
        """Test creation of dummy training data"""
        pipeline = AURATrainingPipeline(self.config)
        dummy_data = pipeline._create_dummy_data(50)
        
        self.assertEqual(len(dummy_data), 50)
        self.assertIn('prompt', dummy_data[0])
        self.assertIn('response', dummy_data[0])
        self.assertIn('consciousness_context', dummy_data[0])
    
    def test_phase0_training(self):
        """Test Phase 0 core initialization training"""
        pipeline = AURATrainingPipeline(self.config)
        training_state = pipeline.phase0_core_initialization()
        
        # Check that metrics were recorded
        self.assertGreater(len(pipeline.metrics['phase0']['loss']), 0)
    
    def test_phase1_training(self):
        """Test Phase 1 consciousness integration training"""
        pipeline = AURATrainingPipeline(self.config)
        training_state = pipeline.setup_training_state(self.config.phase1_learning_rate)
        training_state = pipeline.phase1_consciousness_integration(training_state)
        
        # Check that metrics were recorded
        self.assertGreater(len(pipeline.metrics['phase1']['loss']), 0)
        self.assertGreater(len(pipeline.metrics['phase1']['consciousness_level']), 0)
    
    def test_phase2_training(self):
        """Test Phase 2 self-teaching refinement training"""
        pipeline = AURATrainingPipeline(self.config)
        training_state = pipeline.setup_training_state(self.config.phase2_learning_rate)
        training_state = pipeline.phase2_self_teaching_refinement(training_state)
        
        # Check that metrics were recorded
        self.assertGreater(len(pipeline.metrics['phase2']['loss']), 0)
    
    def test_full_pipeline_execution(self):
        """Test full training pipeline execution"""
        pipeline = AURATrainingPipeline(self.config)
        results = pipeline.run_full_training_pipeline()
        
        # Check results structure
        self.assertIn('training_duration', results)
        self.assertIn('metrics', results)
        self.assertIn('phases_completed', results)
        self.assertEqual(results['phases_completed'], ['phase0', 'phase1', 'phase2'])
        
        # Check that all phases recorded metrics
        self.assertGreater(len(results['metrics']['phase0']['loss']), 0)
        self.assertGreater(len(results['metrics']['phase1']['loss']), 0)
        self.assertGreater(len(results['metrics']['phase2']['loss']), 0)
    
    def test_results_export(self):
        """Test training results export functionality"""
        pipeline = AURATrainingPipeline(self.config)
        results = pipeline.run_full_training_pipeline()
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            pipeline.export_training_results(results, temp_path)
            
            # Verify file was created and contains valid JSON
            with open(temp_path, 'r') as f:
                exported_results = json.load(f)
            
            self.assertIn('training_duration', exported_results)
            self.assertIn('metrics', exported_results)
        finally:
            # Clean up temporary file
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
