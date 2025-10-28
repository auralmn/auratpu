#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import jax
import jax.numpy as jnp


@dataclass
class GlobalWorkspaceConfig:
    capacity: int = 7  # Miller's magic number
    threshold: float = 0.6  # Consciousness threshold


@dataclass
class ConsciousContent:
    """Represents a conscious content candidate"""
    id: str
    text: str
    embedding: jnp.ndarray
    timestamp: float
    strength: float = 1.0
    relevance: float = 0.0
    novelty: float = 0.0
    emotional_salience: float = 0.0
    
    def total_score(self) -> float:
        """
        Calculate total consciousness score based on multiple criteria.
        Weighted combination of metrics with emphasis on relevance and novelty.
        """
        # Weighted scoring: 40% relevance, 30% novelty, 20% emotional_salience, 10% strength
        score = (0.4 * self.relevance + 
                0.3 * self.novelty + 
                0.2 * self.emotional_salience + 
                0.1 * self.strength)
        
        # Apply strength as a multiplier
        score *= self.strength
        
        return score


class GlobalWorkspaceManager:
    """
    Implements Global Workspace Theory for consciousness system.
    Coordinates consciousness competition and maintains conscious contents.
    """
    
    def __init__(self, config: GlobalWorkspaceConfig):
        self.config = config
        self.workspace: List[ConsciousContent] = []
    
    def compete_for_consciousness(self, candidates: List[ConsciousContent]) -> List[ConsciousContent]:
        """
        Implement consciousness competition with multi-criteria evaluation.
        
        Args:
            candidates: List of ConsciousContent candidates for workspace access
            
        Returns:
            List of ConsciousContent winners that gained consciousness access
        """
        # Score all candidates
        scored_candidates = []
        for candidate in candidates:
            score = candidate.total_score()
            if score >= self.config.threshold:
                scored_candidates.append((score, candidate))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Select winners up to workspace capacity
        winners = []
        for score, candidate in scored_candidates:
            if len(self.workspace) < self.config.capacity:
                self.workspace.append(candidate)
                winners.append(candidate)
        
        return winners
    
    def get_workspace_status(self) -> Dict[str, Any]:
        """
        Get current workspace status for monitoring.
        
        Returns:
            Dictionary with workspace information
        """
        return {
            'current_size': len(self.workspace),
            'capacity': self.config.capacity,
            'consciousness_threshold': self.config.threshold,
            'contents': [
                {
                    'id': content.id,
                    'text': content.text,
                    'score': content.total_score(),
                    'timestamp': content.timestamp
                }
                for content in self.workspace
            ]
        }
    
    def clear_workspace(self):
        """Clear all contents from workspace."""
        self.workspace = []
    
    def remove_oldest_content(self):
        """Remove the oldest content from workspace to make room."""
        if self.workspace:
            # Sort by timestamp and remove oldest
            self.workspace.sort(key=lambda x: x.timestamp)
            self.workspace.pop(0)
