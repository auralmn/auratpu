#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class TokenDecoder(nn.Module):
    """
    Token decoder for self-teaching LLM.
    Maps spiking rates to vocabulary logits and samples tokens with phasic response generation.
    """
    hidden_dim: int
    vocab_size: int
    
    # Phasic response parameters
    phase_offset: float = 0.0  # Stylistic phase offset
    frequency_bias: float = 0.0  # Linguistic rhythm bias
    
    def setup(self):
        # Linear readout layer
        self.fc = nn.Dense(self.vocab_size)
        
        # Phase trajectory tracking
        self.phase_trajectory = self.variable('state', 'phase_trajectory', jnp.zeros, (self.hidden_dim,))
    
    def __call__(self, rate_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Decode spiking rates to token probabilities.
        
        Args:
            rate_vector: Spiking rate vector [batch, hidden_dim]
            
        Returns:
            Token probabilities [batch, vocab_size]
        """
        # Update phase trajectory
        if self.phase_trajectory.is_mutable():
            # Simple phase update based on rate vector
            phase_update = jnp.mean(rate_vector, axis=0)  # [hidden_dim]
            new_phase = self.phase_trajectory.value + phase_update
            self.phase_trajectory.value = new_phase
        
        # Apply phase modulation to rate vector
        # This simulates stylistic tone or personality
        phase_mod = jnp.sin(self.phase_trajectory.value + self.phase_offset)
        modulated_rates = rate_vector * (1 + self.frequency_bias * phase_mod)
        
        # Map firing-rate vector to vocabulary logits
        logits = self.fc(modulated_rates)  # [batch, vocab_size]
        
        # Convert logits to probabilities
        probs = jax.nn.softmax(logits, axis=-1)  # [batch, vocab_size]
        
        return probs
    
    def sample_tokens(self, rate_vector: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Sample tokens from rate vector with phasic response generation.
        
        Args:
            rate_vector: Spiking rate vector [batch, hidden_dim]
            key: JAX random key
            
        Returns:
            Sampled token indices [batch]
        """
        # Get probabilities
        token_probs = self.__call__(rate_vector)  # [batch, vocab_size]
        
        # Sample tokens
        token_indices = jax.random.categorical(key, token_probs)  # [batch]
        
        return token_indices
    
    def set_response_mode(self, mode: str):
        """Set response generation mode for phasic modulation."""
        mode_params = {
            'analytical': (0, 0.1),      # Low frequency bias
            'poetic': (jnp.pi/2, 0.5),   # Medium frequency bias
            'assertive': (jnp.pi, 0.8),  # High frequency bias
            'empathetic': (3*jnp.pi/2, 0.3)  # Low-medium frequency bias
        }
        
        if mode in mode_params:
            self.phase_offset, self.frequency_bias = mode_params[mode]
        else:
            # Default to analytical mode
            self.phase_offset, self.frequency_bias = mode_params['analytical']
