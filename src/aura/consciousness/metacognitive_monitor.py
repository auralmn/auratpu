#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import collections
import jax
import jax.numpy as jnp
from typing import List, Deque
from dataclasses import dataclass


@dataclass
class MetacognitiveConfig:
    history_window: int = 100  # Number of recent samples to track
    calibration_window: int = 50  # Window for confidence calibration


class MetacognitiveMonitor:
    """
    Implements metacognitive monitoring for consciousness system.
    Tracks self-awareness through performance consistency and confidence calibration.
    """
    
    def __init__(self, config: MetacognitiveConfig):
        self.config = config
        self.performance_history: Deque[float] = collections.deque(maxlen=config.history_window)
        self.confidence_scores: Deque[float] = collections.deque(maxlen=config.history_window)
        self.self_awareness_level: float = 0.0
        self.calibration_error: float = 0.0
    
    def update_performance(self, performance: float, confidence: float):
        """
        Update performance and confidence tracking.
        
        Args:
            performance: Task performance score (0.0 to 1.0)
            confidence: Confidence in performance (0.0 to 1.0)
        """
        self.performance_history.append(performance)
        self.confidence_scores.append(confidence)
        
        # Update self-awareness metrics
        self._update_self_awareness()
    
    def _update_self_awareness(self):
        """Update self-awareness level based on performance consistency and calibration."""
        if len(self.performance_history) < 10:
            return
        
        # Convert to JAX arrays for computation
        performance_array = jnp.array(list(self.performance_history))
        confidence_array = jnp.array(list(self.confidence_scores))
        
        # Calculate performance consistency (inverse of standard deviation)
        performance_std = jnp.std(performance_array)
        consistency_score = 1.0 / (1.0 + performance_std)
        
        # Calculate confidence calibration (inverse of expected calibration error)
        self.calibration_error = self._calculate_expected_calibration_error(
            performance_array, confidence_array)
        calibration_score = 1.0 - self.calibration_error
        
        # Combine metrics for overall self-awareness
        self.self_awareness_level = float((consistency_score + calibration_score) / 2.0)
    
    def _calculate_expected_calibration_error(self, performance: jnp.ndarray, 
                                            confidence: jnp.ndarray) -> float:
        """
        Calculate expected calibration error (ECE) between performance and confidence.
        
        Args:
            performance: Array of performance scores
            confidence: Array of confidence scores
            
        Returns:
            Expected calibration error (0.0 to 1.0)
        """
        # Create bins for confidence scores
        n_bins = 10
        bin_boundaries = jnp.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        total_samples = performance.shape[0]
        
        for i in range(n_bins):
            # Find samples in this confidence bin
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # For the last bin, include upper boundary
            if i == n_bins - 1:
                in_bin = (confidence >= bin_lower) & (confidence <= bin_upper)
            else:
                in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
            
            bin_size = jnp.sum(in_bin)
            
            if bin_size > 0:
                # Calculate average performance and confidence in bin
                avg_performance = jnp.mean(performance[in_bin])
                avg_confidence = jnp.mean(confidence[in_bin])
                
                # Calculate calibration error for this bin
                bin_error = jnp.abs(avg_confidence - avg_performance)
                
                # Weight by bin size
                ece += (bin_size / total_samples) * bin_error
        
        return float(ece)
    
    def get_metacognitive_status(self) -> dict:
        """
        Get current metacognitive status for monitoring.
        
        Returns:
            Dictionary with metacognitive metrics
        """
        return {
            'self_awareness_level': self.self_awareness_level,
            'calibration_error': self.calibration_error,
            'performance_history_length': len(self.performance_history),
            'confidence_history_length': len(self.confidence_scores),
            'average_performance': float(jnp.mean(jnp.array(list(self.performance_history)))) if self.performance_history else 0.0,
            'average_confidence': float(jnp.mean(jnp.array(list(self.confidence_scores)))) if self.confidence_scores else 0.0
        }
    
    def adjust_thresholds(self) -> dict:
        """
        Adjust consciousness thresholds based on self-awareness level.
        
        Returns:
            Dictionary with adjusted thresholds
        """
        # Base thresholds from self-awareness level
        consciousness_threshold = 0.5 + 0.3 * self.self_awareness_level  # Range: 0.5 to 0.8
        novelty_threshold = 0.6 + 0.2 * self.self_awareness_level       # Range: 0.6 to 0.8
        
        return {
            'consciousness_threshold': float(consciousness_threshold),
            'novelty_threshold': float(novelty_threshold)
        }
