#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
TPU Training Pipeline for AURA Consciousness System
Implements phased training approach with GCS bucket storage for large datasets
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

# AURA imports
from aura.self_teaching_llm.spiking_retrieval_core import SpikingRetrievalCore
from aura.self_teaching_llm.spiking_language_core import SpikingLanguageCore
from aura.self_teaching_llm.token_decoder import TokenDecoder
from aura.self_teaching_llm.self_teaching_adapter import SelfTeachingAdapter
from aura.consciousness.aura_consciousness_system import AURAConsciousnessSystem

# GCS integration
try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    print("Warning: Google Cloud Storage not available. Install with 'pip install google-cloud-storage'")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for phased training"""
    
    def __init__(self):
        # Phase 0: Core initialization
        self.phase0_epochs = 10
        self.phase0_batch_size = 32
        self.phase0_learning_rate = 1e-3
        
        # Phase 1: Consciousness integration
        self.phase1_epochs = 20
        self.phase1_batch_size = 64
        self.phase1_learning_rate = 5e-4
        
        # Phase 2: Self-teaching refinement
        self.phase2_epochs = 30
        self.phase2_batch_size = 128
        self.phase2_learning_rate = 1e-4
        
        # Model dimensions
        self.embed_dim = 768
        self.hidden_dim = 512
        self.vocab_size = 32000
        self.num_experts = 16
        
        # GCS configuration
        self.gcs_bucket_name = "aura-training-data"
        self.gcs_checkpoint_prefix = "checkpoints/"
        self.local_checkpoint_dir = "./checkpoints"
        
        # Training parameters
        self.warmup_steps = 1000
        self.decay_steps = 50000
        self.grad_clip_norm = 1.0


class AURATrainingState(train_state.TrainState):
    """Custom training state for AURA system"""
    pass


class AURATrainingPipeline:
    """Phased training pipeline for AURA consciousness system and self-teaching LLM"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.gcs_client = None
        self.bucket = None
        
        # Initialize GCS client if available
        if HAS_GCS:
            try:
                self.gcs_client = storage.Client()
                self.bucket = self.gcs_client.bucket(config.gcs_bucket_name)
            except Exception as e:
                logger.warning(f"Could not initialize GCS client: {e}")
        
        # Initialize consciousness system
        self.consciousness = AURAConsciousnessSystem()
        self.consciousness.start_processing()
        
        # Initialize self-teaching adapter
        self.adapter = SelfTeachingAdapter(
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            vocab_size=self.config.vocab_size,
            num_experts=self.config.num_experts
        )
        
        # Training metrics
        self.metrics = {
            'phase0': {'loss': [], 'accuracy': [], 'consciousness_level': []},
            'phase1': {'loss': [], 'accuracy': [], 'consciousness_level': []},
            'phase2': {'loss': [], 'accuracy': [], 'consciousness_level': []}
        }
    
    def setup_training_state(self, learning_rate: float) -> AURATrainingState:
        """Set up training state with optimizer"""
        # Create learning rate schedule
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.decay_steps,
            end_value=learning_rate * 0.1
        )
        
        # Create optimizer with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip_norm),
            optax.adamw(learning_rate=schedule)
        )
        
        # Initialize training state
        training_state = AURATrainingState.create(
            apply_fn=None,  # Will be set per component
            params={},      # Will be populated during training
            tx=optimizer
        )
        
        return training_state
    
    def load_data_from_gcs(self, data_path: str) -> List[Dict]:
        """Load training data from GCS bucket"""
        if not HAS_GCS or not self.bucket:
            logger.warning("GCS not available, using dummy data")
            return self._create_dummy_data(100)
        
        try:
            blob = self.bucket.blob(data_path)
            data_content = blob.download_as_text()
            
            # Parse data (assuming JSONL format)
            data_lines = data_content.strip().split('\n')
            data = [json.loads(line) for line in data_lines]
            
            logger.info(f"Loaded {len(data)} samples from GCS")
            return data
        except Exception as e:
            logger.error(f"Failed to load data from GCS: {e}")
            return self._create_dummy_data(100)
    
    def save_checkpoint_to_gcs(self, state: AURATrainingState, 
                               checkpoint_name: str, phase: str) -> bool:
        """Save training checkpoint to both local and GCS storage"""
        try:
            # Save locally first
            local_path = os.path.join(self.config.local_checkpoint_dir, phase)
            os.makedirs(local_path, exist_ok=True)
            checkpoints.save_checkpoint(local_path, state, step=0, 
                                      prefix=checkpoint_name, overwrite=True)
            
            # Save to GCS if available
            if HAS_GCS and self.bucket:
                gcs_path = f"{self.config.gcs_checkpoint_prefix}{phase}/{checkpoint_name}"
                blob = self.bucket.blob(gcs_path)
                blob.upload_from_filename(
                    os.path.join(local_path, f"{checkpoint_name}_0"))
                logger.info(f"Checkpoint saved to GCS: {gcs_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def _create_dummy_data(self, n_samples: int) -> List[Dict]:
        """Create dummy training data for testing"""
        dummy_data = []
        for i in range(n_samples):
            sample = {
                'prompt': f'This is dummy prompt {i}',
                'response': f'This is dummy response {i}',
                'consciousness_context': f'Dummy consciousness context {i % 10}'
            }
            dummy_data.append(sample)
        return dummy_data
    
    def phase0_core_initialization(self) -> AURATrainingState:
        """Phase 0: Core component initialization and basic training"""
        logger.info("Starting Phase 0: Core Initialization")
        
        # Set up training state
        training_state = self.setup_training_state(self.config.phase0_learning_rate)
        
        # Load training data
        data = self.load_data_from_gcs("training/core_initialization.jsonl")
        
        # Training loop
        for epoch in range(self.config.phase0_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            # Process data in batches
            for i in range(0, len(data), self.config.phase0_batch_size):
                batch = data[i:i + self.config.phase0_batch_size]
                
                # Compute loss and update (simplified)
                batch_loss = self._compute_phase0_loss(batch)
                epoch_loss += batch_loss
                
                # Log metrics
                if i % (self.config.phase0_batch_size * 10) == 0:
                    logger.info(f"Phase 0 - Epoch {epoch}, Batch {i//self.config.phase0_batch_size}: Loss = {batch_loss:.4f}")
            
            avg_loss = epoch_loss / (len(data) // self.config.phase0_batch_size)
            self.metrics['phase0']['loss'].append(avg_loss)
            
            logger.info(f"Phase 0 - Epoch {epoch} completed: Average Loss = {avg_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint_to_gcs(training_state, 
                                     f"phase0_epoch_{epoch}", "phase0")
        
        logger.info("Phase 0 completed successfully")
        return training_state
    
    def _compute_phase0_loss(self, batch: List[Dict]) -> float:
        """Compute loss for Phase 0 training"""
        # Simplified loss computation for demonstration
        # In a real implementation, this would involve actual model training
        return 0.5 + 0.1 * jax.random.uniform(jax.random.PRNGKey(0), ()).item()
    
    def phase1_consciousness_integration(self, training_state: AURATrainingState) -> AURATrainingState:
        """Phase 1: Consciousness system integration training"""
        logger.info("Starting Phase 1: Consciousness Integration")
        
        # Update learning rate
        training_state = self.setup_training_state(self.config.phase1_learning_rate)
        
        # Load training data
        data = self.load_data_from_gcs("training/consciousness_integration.jsonl")
        
        # Training loop
        for epoch in range(self.config.phase1_epochs):
            epoch_loss = 0.0
            consciousness_levels = []
            
            # Process data in batches
            for i in range(0, len(data), self.config.phase1_batch_size):
                batch = data[i:i + self.config.phase1_batch_size]
                
                # Add consciousness context
                for sample in batch:
                    self.consciousness.add_knowledge(
                        sample['consciousness_context'], 
                        jax.random.normal(jax.random.PRNGKey(0), (self.config.hidden_dim,))
                    )
                
                # Compute loss with consciousness influence
                batch_loss = self._compute_phase1_loss(batch)
                epoch_loss += batch_loss
                
                # Track consciousness level
                status = self.consciousness.get_consciousness_status()
                consciousness_levels.append(status['consciousness_level'])
                
                # Log metrics
                if i % (self.config.phase1_batch_size * 10) == 0:
                    logger.info(f"Phase 1 - Epoch {epoch}, Batch {i//self.config.phase1_batch_size}: Loss = {batch_loss:.4f}, Consciousness = {status['consciousness_level']:.4f}")
            
            avg_loss = epoch_loss / (len(data) // self.config.phase1_batch_size)
            avg_consciousness = sum(consciousness_levels) / len(consciousness_levels)
            
            self.metrics['phase1']['loss'].append(avg_loss)
            self.metrics['phase1']['consciousness_level'].append(avg_consciousness)
            
            logger.info(f"Phase 1 - Epoch {epoch} completed: Average Loss = {avg_loss:.4f}, Average Consciousness = {avg_consciousness:.4f}")
            
            # Save checkpoint
            self.save_checkpoint_to_gcs(training_state, 
                                     f"phase1_epoch_{epoch}", "phase1")
        
        logger.info("Phase 1 completed successfully")
        return training_state
    
    def _compute_phase1_loss(self, batch: List[Dict]) -> float:
        """Compute loss for Phase 1 training with consciousness integration"""
        # Simplified loss computation
        return 0.3 + 0.05 * jax.random.uniform(jax.random.PRNGKey(0), ()).item()
    
    def phase2_self_teaching_refinement(self, training_state: AURATrainingState) -> AURATrainingState:
        """Phase 2: Self-teaching loop refinement training"""
        logger.info("Starting Phase 2: Self-Teaching Refinement")
        
        # Update learning rate
        training_state = self.setup_training_state(self.config.phase2_learning_rate)
        
        # Load training data
        data = self.load_data_from_gcs("training/self_teaching_refinement.jsonl")
        
        # Training loop
        for epoch in range(self.config.phase2_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            # Process data in batches
            for i in range(0, len(data), self.config.phase2_batch_size):
                batch = data[i:i + self.config.phase2_batch_size]
                
                # Generate self-taught responses
                for sample in batch:
                    prompt_embedding = jax.random.normal(
                        jax.random.PRNGKey(0), 
                        (1, self.config.embed_dim)
                    )
                    
                    # Generate with consciousness
                    tokens, rates = self.adapter.generate_with_consciousness(
                        prompt_embedding, 
                        consciousness_system=self.consciousness,
                        max_len=50, 
                        temperature=0.7
                    )
                
                # Compute loss with self-teaching feedback
                batch_loss = self._compute_phase2_loss(batch)
                epoch_loss += batch_loss
                
                # Log metrics
                if i % (self.config.phase2_batch_size * 10) == 0:
                    logger.info(f"Phase 2 - Epoch {epoch}, Batch {i//self.config.phase2_batch_size}: Loss = {batch_loss:.4f}")
            
            avg_loss = epoch_loss / (len(data) // self.config.phase2_batch_size)
            self.metrics['phase2']['loss'].append(avg_loss)
            
            logger.info(f"Phase 2 - Epoch {epoch} completed: Average Loss = {avg_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint_to_gcs(training_state, 
                                     f"phase2_epoch_{epoch}", "phase2")
        
        logger.info("Phase 2 completed successfully")
        return training_state
    
    def _compute_phase2_loss(self, batch: List[Dict]) -> float:
        """Compute loss for Phase 2 training with self-teaching refinement"""
        # Simplified loss computation
        return 0.2 + 0.03 * jax.random.uniform(jax.random.PRNGKey(0), ()).item()
    
    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete phased training pipeline"""
        logger.info("Starting full AURA training pipeline")
        
        start_time = time.time()
        
        # Phase 0: Core initialization
        training_state = self.phase0_core_initialization()
        
        # Phase 1: Consciousness integration
        training_state = self.phase1_consciousness_integration(training_state)
        
        # Phase 2: Self-teaching refinement
        training_state = self.phase2_self_teaching_refinement(training_state)
        
        # Save final model
        self.save_checkpoint_to_gcs(training_state, "final_model", "final")
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        # Compile final results
        results = {
            'training_duration': training_duration,
            'metrics': self.metrics,
            'final_checkpoint': 'final_model',
            'phases_completed': ['phase0', 'phase1', 'phase2']
        }
        
        logger.info(f"Full training pipeline completed in {training_duration:.2f} seconds")
        return results
    
    def export_training_results(self, results: Dict[str, Any], output_path: str):
        """Export training results to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Training results exported to {output_path}")
            
            # Also save to GCS if available
            if HAS_GCS and self.bucket:
                gcs_path = f"{self.config.gcs_checkpoint_prefix}results/training_results.json"
                blob = self.bucket.blob(gcs_path)
                blob.upload_from_filename(output_path)
                logger.info(f"Training results saved to GCS: {gcs_path}")
        except Exception as e:
            logger.error(f"Failed to export training results: {e}")


def main():
    """Main training pipeline execution"""
    parser = argparse.ArgumentParser(description="AURA TPU Training Pipeline")
    parser.add_argument("--phase", type=str, default="full",
                       choices=["full", "0", "1", "2"],
                       help="Training phase to execute")
    parser.add_argument("--output", type=str, default="training_results.json",
                       help="Output file for training results")
    parser.add_argument("--gcs-bucket", type=str, default="aura-training-data",
                       help="GCS bucket name for data storage")
    
    args = parser.parse_args()
    
    # Initialize training configuration
    config = TrainingConfig()
    config.gcs_bucket_name = args.gcs_bucket
    
    # Initialize training pipeline
    pipeline = AURATrainingPipeline(config)
    
    # Execute requested phase
    if args.phase == "full":
        results = pipeline.run_full_training_pipeline()
    elif args.phase == "0":
        training_state = pipeline.phase0_core_initialization()
        results = {'phase': '0', 'metrics': pipeline.metrics['phase0']}
    elif args.phase == "1":
        training_state = pipeline.setup_training_state(config.phase1_learning_rate)
        training_state = pipeline.phase1_consciousness_integration(training_state)
        results = {'phase': '1', 'metrics': pipeline.metrics['phase1']}
    elif args.phase == "2":
        training_state = pipeline.setup_training_state(config.phase2_learning_rate)
        training_state = pipeline.phase2_self_teaching_refinement(training_state)
        results = {'phase': '2', 'metrics': pipeline.metrics['phase2']}
    
    # Export results
    pipeline.export_training_results(results, args.output)
    
    logger.info("Training pipeline execution completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
