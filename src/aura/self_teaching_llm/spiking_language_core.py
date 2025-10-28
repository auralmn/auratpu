#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class SpikingLanguageCore(nn.Module):
    """
    Spiking language core for self-teaching LLM.
    Processes context vectors with biologically plausible spiking dynamics.
    """
    hidden_dim: int
    dt: float = 1e-3  # Time step
    T: int = 20  # Number of time steps for temporal simulation
    tau: float = 20e-3  # Membrane time constant
    v_th: float = 1.0  # Spike threshold
    v_reset: float = 0.0  # Reset voltage
    
    def setup(self):
        # Recurrent weights
        self.recurrent_weights = self.param('recurrent_weights', 
                                         nn.initializers.normal(stddev=0.1),
                                         (self.hidden_dim, self.hidden_dim))
        
        # Input projection to match dimensions
        self.input_projection = nn.Dense(self.hidden_dim)
    
    def __call__(self, input_state: jnp.ndarray, prev_state: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Process input through spiking language core.
        
        Args:
            input_state: Input context vector [batch, hidden_dim]
            prev_state: Previous state tuple (voltage, spike) [batch, hidden_dim] each
            
        Returns:
            Tuple of (output_rate, next_state)
        """
        prev_voltage, prev_spike = prev_state
        
        # Project input to match hidden dimensions
        projected_input = self.input_projection(input_state)  # [batch, hidden_dim]
        
        # Initialize membrane potentials
        v = prev_voltage  # [batch, hidden_dim]
        s = prev_spike    # [batch, hidden_dim]
        
        # Simulate spiking dynamics over time
        spike_accum = jnp.zeros_like(s)
        
        for t in range(self.T):
            # Compute membrane potential update
            # dV/dt = (-V + W*S + I) / tau
            recurrent_input = jnp.dot(s, self.recurrent_weights)  # [batch, hidden_dim]
            
            # Membrane potential dynamics
            dv = (-v + recurrent_input + projected_input) / self.tau * self.dt
            v = v + dv
            
            # Spike generation (Heaviside step function)
            spike = (v >= self.v_th).astype(jnp.float32)
            
            # Voltage reset after spike
            v = v * (1 - spike) + self.v_reset * spike
            
            # Accumulate spikes
            spike_accum = spike_accum + spike
            
            # Update spike state
            s = spike
        
        # Average spikes over time steps
        avg_spikes = spike_accum / self.T  # [batch, hidden_dim]
        
        # Return output rate and updated state
        next_state = (v, s)
        return avg_spikes, next_state
    
    def initialize_state(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize recurrent state for spiking core.
        
        Args:
            batch_size: Batch size for state initialization
            
        Returns:
            Tuple of (voltage, spike) initialized to zeros
        """
        voltage = jnp.zeros((batch_size, self.hidden_dim))
        spike = jnp.zeros((batch_size, self.hidden_dim))
        return (voltage, spike)
