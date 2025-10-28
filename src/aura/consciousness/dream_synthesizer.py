#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import jax
import jax.numpy as jnp
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .global_workspace_manager import ConsciousContent


@dataclass
class DreamSynthesizerConfig:
    novelty_threshold: float = 0.7
    max_synthesis_attempts: int = 10
    synthesis_fragment_count: int = 3


class AURADreamSynthesizer:
    """
    Implements dream synthesis with theta-gamma coupling and hyperassociative recombination.
    Generates novel conscious contents during REM sleep-like states.
    """
    
    def __init__(self, config: DreamSynthesizerConfig):
        self.config = config
    
    def synthesize_concepts(self, fragments: List[Dict[str, Any]], 
                          gamma_amplitude: float) -> List[ConsciousContent]:
        """
        Synthesize novel concepts using hyperassociative recombination during coupling windows.
        
        Args:
            fragments: List of memory fragments to use for synthesis
            gamma_amplitude: Current gamma amplitude for modulation
            
        Returns:
            List of synthesized ConsciousContent candidates
        """
        synthesized = []
        
        # Limit synthesis attempts to prevent infinite loops
        for i in range(min(len(fragments), self.config.max_synthesis_attempts)):
            # Select diverse fragments for recombination
            selected_fragments = self._select_diverse_fragments(fragments, 
                                                              self.config.synthesis_fragment_count)
            
            if not selected_fragments:
                continue
                
            # Perform activation-weighted combination
            synthesis_vector = self._combine_fragments(selected_fragments, gamma_amplitude)
            
            # Calculate novelty score compared to existing fragments
            novelty_score = self._calculate_novelty(synthesis_vector, selected_fragments)
            
            # Only create content if novelty exceeds threshold
            if novelty_score > self.config.novelty_threshold:
                content = self._create_conscious_content(selected_fragments, 
                                                       synthesis_vector, 
                                                       novelty_score)
                synthesized.append(content)
        
        return synthesized
    
    def _select_diverse_fragments(self, fragments: List[Dict[str, Any]], 
                                count: int) -> List[Dict[str, Any]]:
        """
        Select diverse fragments for recombination to maximize novelty.
        
        Args:
            fragments: List of available fragments
            count: Number of fragments to select
            
        Returns:
            List of selected diverse fragments
        """
        if len(fragments) < count:
            return []
        
        # Start with a random fragment
        selected = [random.choice(fragments)]
        
        # Iteratively select most dissimilar fragments
        for _ in range(count - 1):
            if not fragments:
                break
                
            # Find fragment with minimum similarity to already selected ones
            best_fragment = None
            min_similarity = float('inf')
            
            for fragment in fragments:
                if fragment in selected:
                    continue
                    
                # Calculate average similarity to selected fragments
                total_similarity = 0.0
                for selected_fragment in selected:
                    similarity = self._calculate_similarity(fragment['embedding'], 
                                                         selected_fragment['embedding'])
                    total_similarity += similarity
                
                avg_similarity = total_similarity / len(selected)
                
                if avg_similarity < min_similarity:
                    min_similarity = avg_similarity
                    best_fragment = fragment
            
            if best_fragment:
                selected.append(best_fragment)
        
        return selected
    
    def _combine_fragments(self, fragments: List[Dict[str, Any]], 
                          gamma_amplitude: float) -> jnp.ndarray:
        """
        Perform activation-weighted combination of fragment embeddings.
        
        Args:
            fragments: List of fragments to combine
            gamma_amplitude: Gamma amplitude modulation factor
            
        Returns:
            Combined synthesis vector
        """
        if not fragments:
            return jnp.zeros(0)
        
        # Get embedding dimension from first fragment
        embedding_dim = fragments[0]['embedding'].shape[0]
        synthesis_vector = jnp.zeros(embedding_dim)
        
        # Weighted combination based on fragment strength and gamma amplitude
        total_weight = 0.0
        for fragment in fragments:
            strength = fragment.get('strength', 1.0)
            weight = strength * gamma_amplitude
            synthesis_vector += weight * fragment['embedding']
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            synthesis_vector = synthesis_vector / total_weight
        
        return synthesis_vector
    
    def _calculate_novelty(self, synthesis_vector: jnp.ndarray, 
                          selected_fragments: List[Dict[str, Any]]) -> float:
        """
        Calculate novelty score based on cosine similarity to selected fragments.
        
        Args:
            synthesis_vector: The synthesized embedding vector
            selected_fragments: Fragments used in synthesis
            
        Returns:
            Novelty score (lower similarity = higher novelty)
        """
        if not selected_fragments or synthesis_vector.shape[0] == 0:
            return 0.0
        
        # Calculate average similarity to selected fragments
        total_similarity = 0.0
        for fragment in selected_fragments:
            similarity = self._calculate_similarity(synthesis_vector, fragment['embedding'])
            total_similarity += similarity
        
        avg_similarity = total_similarity / len(selected_fragments)
        
        # Convert similarity to novelty (1 - similarity)
        novelty_score = 1.0 - avg_similarity
        
        return novelty_score
    
    def _calculate_similarity(self, vec1: jnp.ndarray, vec2: jnp.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if vec1.shape[0] == 0 or vec2.shape[0] == 0:
            return 0.0
        
        # Normalize vectors
        norm1 = jnp.linalg.norm(vec1)
        norm2 = jnp.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        normalized_vec1 = vec1 / norm1
        normalized_vec2 = vec2 / norm2
        
        # Calculate cosine similarity
        similarity = jnp.dot(normalized_vec1, normalized_vec2)
        
        # Clamp to [0, 1] range
        return float(jnp.clip(similarity, 0.0, 1.0))
    
    def _create_conscious_content(self, fragments: List[Dict[str, Any]], 
                                synthesis_vector: jnp.ndarray, 
                                novelty_score: float) -> ConsciousContent:
        """
        Create a new ConsciousContent from synthesized fragments.
        
        Args:
            fragments: Source fragments for synthesis
            synthesis_vector: Combined embedding vector
            novelty_score: Calculated novelty score
            
        Returns:
            New ConsciousContent object
        """
        # Combine fragment texts into synthesized content text
        fragment_texts = [fragment['text'] for fragment in fragments]
        combined_text = " ".join(fragment_texts)
        
        # Create unique ID for synthesized content
        fragment_ids = [fragment['id'] for fragment in fragments]
        synthesis_id = f"synth_{'_'.join(fragment_ids)}"
        
        # Calculate average timestamp of source fragments
        avg_timestamp = sum(fragment['timestamp'] for fragment in fragments) / len(fragments)
        
        # Calculate average strength of source fragments
        avg_strength = sum(fragment.get('strength', 1.0) for fragment in fragments) / len(fragments)
        
        return ConsciousContent(
            id=synthesis_id,
            text=combined_text,
            embedding=synthesis_vector,
            timestamp=avg_timestamp,
            strength=avg_strength,
            novelty=novelty_score,
            # Set default values for other attributes
            relevance=0.5,  # Default relevance
            emotional_salience=0.5  # Default emotional salience
        )
