#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class OscillatorConfig:
    theta_freq: float = 6.0  # Hz
    gamma_freq: float = 40.0  # Hz


class ThetaGammaOscillator:
    """
    Biologically-inspired theta-gamma oscillator for consciousness system.
    Theta (4-8 Hz) provides temporal windows for memory processing.
    Gamma (30-100 Hz) encodes fine-grained information.
    """
    
    def __init__(self, config: OscillatorConfig):
        self.config = config
        self.theta_phase: float = 0.0
        self.gamma_phase: float = 0.0
        self.last_update: float = time.time()
    
    def update_phases(self) -> Tuple[float, float]:
        """Update oscillator phases based on elapsed time."""
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        
        # Update phases (radians)
        self.theta_phase += 2 * math.pi * self.config.theta_freq * dt
        self.gamma_phase += 2 * math.pi * self.config.gamma_freq * dt
        
        # Keep phases in [0, 2π] range
        self.theta_phase %= 2 * math.pi
        self.gamma_phase %= 2 * math.pi
        
        return self.theta_phase, self.gamma_phase
    
    def get_coupling_window(self) -> bool:
        """
        Determine if we're in a theta-gamma coupling window.
        Coupling occurs when theta phase is near peak (sin > 0.9).
        """
        return abs(math.sin(self.theta_phase)) > 0.9
    
    def get_gamma_amplitude(self) -> float:
        """
        Get gamma amplitude modulated by theta phase.
        This implements cross-frequency coupling where gamma amplitude
        is controlled by theta phase.
        """
        return abs(math.sin(self.gamma_phase)) * (0.5 + 0.5 * math.cos(self.theta_phase))
