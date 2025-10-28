#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Demo script showing how the consciousness system and self-teaching LLM work together.
"""

import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp

from aura.consciousness.aura_consciousness_system import AURAConsciousnessSystem
from aura.self_teaching_llm.self_teaching_adapter import SelfTeachingAdapter


def demo_consciousness_and_llm():
    """Demo the integration between consciousness system and self-teaching LLM."""
    print("=== AURA Consciousness System & Self-Teaching LLM Demo ===")
    
    # Initialize consciousness system
    print("Initializing consciousness system...")
    consciousness = AURAConsciousnessSystem()
    
    # Initialize self-teaching LLM adapter
    print("Initializing self-teaching LLM adapter...")
    llm_adapter = SelfTeachingAdapter(
        embed_dim=64,
        hidden_dim=128,
        vocab_size=1000,
        num_experts=8
    )
    
    # Add some initial knowledge to consciousness system
    print("Adding initial knowledge fragments...")
    initial_knowledge = [
        ("The quick brown fox jumps over the lazy dog", jnp.array([0.8, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])),
        ("Machine learning models process data patterns", jnp.array([0.1, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])),
        ("Neural networks learn through backpropagation", jnp.array([0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("Consciousness emerges from complex neural dynamics", jnp.array([0.1, 0.0, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0])),
        ("Self-teaching systems improve through feedback", jnp.array([0.0, 0.1, 0.0, 0.1, 0.8, 0.0, 0.0, 0.0]))
    ]
    
    for text, embedding in initial_knowledge:
        consciousness.add_knowledge(text, embedding)
    
    # Start consciousness processing
    print("Starting consciousness processing...")
    consciousness.start_processing()
    
    # Let consciousness process for a moment
    time.sleep(0.5)
    
    # Check consciousness status
    print("\n--- Consciousness Status ---")
    status = consciousness.get_consciousness_status()
    print(f"Processing state: {status['processing_state']}")
    print(f"Consciousness level: {status['consciousness_level']:.3f}")
    print(f"Calibration error: {status['calibration_error']:.3f}")
    print(f"Knowledge buffer size: {status['knowledge_buffer_size']}")
    print(f"Workspace contents: {len(status['workspace']['contents'])}")
    
    # Generate text with consciousness context
    print("\n--- Text Generation with Consciousness ---")
    prompt_embedding = jnp.array([[0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]])  # Simple prompt embedding
    
    # Generate without consciousness context
    print("Generating text without consciousness context...")
    tokens_no_context, rates_no_context = llm_adapter.generate_with_consciousness(
        prompt_embedding, max_len=20, temperature=0.8)
    print(f"Generated tokens shape: {tokens_no_context.shape}")
    
    # Generate with consciousness context
    print("Generating text with consciousness context...")
    tokens_with_context, rates_with_context = llm_adapter.generate_with_consciousness(
        prompt_embedding, consciousness_system=consciousness, max_len=20, temperature=0.8)
    print(f"Generated tokens shape: {tokens_with_context.shape}")
    
    # Show that consciousness level affected generation
    print(f"\nConsciousness level influenced generation parameters:")
    adjusted_thresholds = status['adjusted_thresholds']
    print(f"  - Consciousness threshold: {adjusted_thresholds['consciousness_threshold']:.3f}")
    print(f"  - Novelty threshold: {adjusted_thresholds['novelty_threshold']:.3f}")
    
    # Stop consciousness processing
    print("\nStopping consciousness processing...")
    consciousness.stop_processing()
    
    print("\n=== Demo Complete ===")


if __name__ == '__main__':
    demo_consciousness_and_llm()
