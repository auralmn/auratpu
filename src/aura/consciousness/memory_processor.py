#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict, Any

import jax
import jax.numpy as jnp


@dataclass
class MemoryProcessorConfig:
    fragment_size: int = 4
    max_fragments: int = 10000
    perturbation_std: float = 0.01


class AURAMemoryProcessor:
    """
    Processes knowledge into memory fragments for consciousness system.
    Breaks down large-scale knowledge into overlapping semantic chunks
    and applies controlled noise for diversity.
    """
    
    def __init__(self, config: MemoryProcessorConfig):
        self.config = config
    
    def fragment_knowledge(self, text: str, embedding: jnp.ndarray) -> List[Dict[str, Any]]:
        """
        Fragment knowledge into overlapping semantic chunks.
        
        Args:
            text: Input text to fragment
            embedding: Text embedding vector
            
        Returns:
            List of fragment dictionaries with id, text, embedding, strength, timestamp
        """
        words = text.split()
        fragments = []
        
        # Skip if text is too short
        if len(words) < self.config.fragment_size:
            return fragments
        
        for i in range(len(words) - self.config.fragment_size + 1):
            fragment_words = words[i:i + self.config.fragment_size]
            fragment_text = " ".join(fragment_words)
            perturbed_embedding = self._apply_semantic_perturbation(embedding)
            
            fragment = {
                'id': f"frag_{len(fragments)}",
                'text': fragment_text,
                'embedding': perturbed_embedding,
                'strength': 1.0,
                'timestamp': time.time()
            }
            fragments.append(fragment)
        
        return fragments
    
    def _apply_semantic_perturbation(self, embedding: jnp.ndarray) -> jnp.ndarray:
        """
        Apply controlled noise to embedding for diversity.
        
        Args:
            embedding: Original embedding vector
            
        Returns:
            Perturbed embedding vector
        """
        # For reproducibility, use fixed key (in real system would use dynamic key)
        key = jax.random.key(0)
        noise = jax.random.normal(key, embedding.shape) * self.config.perturbation_std
        return embedding + noise
