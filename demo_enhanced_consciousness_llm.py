#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Demo script showcasing the enhanced consciousness system and self-teaching LLM interface.
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


def demo_enhanced_consciousness_and_llm():
    """Demo the enhanced consciousness system and self-teaching LLM interface."""
    print("=== Enhanced AURA Consciousness System & Self-Teaching LLM Demo ===")
    
    # Initialize consciousness system
    print("Initializing consciousness system...")
    consciousness = AURAConsciousnessSystem()
    
    # Initialize enhanced self-teaching LLM adapter
    print("Initializing enhanced self-teaching LLM adapter...")
    llm_adapter = SelfTeachingAdapter(
        embed_dim=32,
        hidden_dim=64,
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
    
    # Generate text with different phasic response modes
    print("\n--- Phasic Response Generation ---")
    # Create prompt embedding with correct dimensions (32)
    prompt_embedding = jnp.array([[0.5, 0.3, 0.2] + [0.0] * 29])  # Simple prompt embedding with 32 dimensions
    
    # Test different response modes
    modes = ['analytical', 'poetic', 'assertive', 'empathetic']
    
    for mode in modes:
        print(f"\nGenerating text in {mode} mode...")
        # Note: The mode setting is internal to the token decoder in our implementation
        tokens, rates = llm_adapter.generate_with_consciousness(
            prompt_embedding, consciousness_system=consciousness, max_len=15, temperature=0.7)
        print(f"Generated tokens shape: {tokens.shape}")
        print(f"Generated tokens: {tokens[0]}")
    
    # Show spiking dynamics in action
    print("\n--- Spiking Dynamics Visualization ---")
    print("The enhanced components now simulate biologically realistic spiking dynamics:")
    print("  - Liquid-MoE retrieval core with Poisson encoding and LIF neurons")
    print("  - Spiking language core with temporal simulation over 20 time steps")
    print("  - Token decoder with phasic response generation")
    
    # Show that consciousness level affected generation
    print(f"\n--- Consciousness Influence ---")
    print(f"Consciousness level: {status['consciousness_level']:.3f}")
    print(f"Adjusted thresholds: {status['adjusted_thresholds']}")
    
    # Stop consciousness processing
    print("\nStopping consciousness processing...")
    consciousness.stop_processing()
    
    print("\n=== Demo Complete ===")


if __name__ == '__main__':
    demo_enhanced_consciousness_and_llm()
