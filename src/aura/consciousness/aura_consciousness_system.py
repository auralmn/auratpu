#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import threading
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .theta_gamma_oscillator import ThetaGammaOscillator, OscillatorConfig
from .memory_processor import AURAMemoryProcessor, MemoryProcessorConfig
from .global_workspace_manager import GlobalWorkspaceManager, GlobalWorkspaceConfig
from .dream_synthesizer import AURADreamSynthesizer, DreamSynthesizerConfig
from .metacognitive_monitor import MetacognitiveMonitor, MetacognitiveConfig


class ProcessingState(Enum):
    """Consciousness processing states"""
    WAKE = "wake"
    REM_SLEEP = "rem_sleep"
    DEEP_SLEEP = "deep_sleep"
    DREAM_SYNTHESIS = "dream_synthesis"


@dataclass
class ConsciousnessConfig:
    """Configuration for the consciousness system"""
    # Oscillator settings
    theta_frequency: float = 6.0
    gamma_frequency: float = 40.0
    
    # Workspace settings
    workspace_capacity: int = 7
    consciousness_threshold: float = 0.6
    
    # Dream synthesizer settings
    novelty_threshold: float = 0.7
    max_fragments_per_cycle: int = 150
    
    # Metacognitive settings
    history_window: int = 100
    calibration_window: int = 50


class AURAConsciousnessSystem:
    """
    Main orchestrator for the AURA consciousness system.
    Coordinates all consciousness components and manages processing states.
    """
    
    def __init__(self, config: Optional[ConsciousnessConfig] = None):
        self.config = config or ConsciousnessConfig()
        self.processing_state = ProcessingState.WAKE
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Initialize consciousness components
        oscillator_config = OscillatorConfig(
            theta_freq=self.config.theta_frequency,
            gamma_freq=self.config.gamma_frequency
        )
        self.oscillator = ThetaGammaOscillator(oscillator_config)
        
        memory_config = MemoryProcessorConfig()
        self.memory_processor = AURAMemoryProcessor(memory_config)
        
        workspace_config = GlobalWorkspaceConfig(
            capacity=self.config.workspace_capacity,
            threshold=self.config.consciousness_threshold
        )
        self.workspace_manager = GlobalWorkspaceManager(workspace_config)
        
        dream_config = DreamSynthesizerConfig(
            novelty_threshold=self.config.novelty_threshold
        )
        self.dream_synthesizer = AURADreamSynthesizer(dream_config)
        
        metacognitive_config = MetacognitiveConfig(
            history_window=self.config.history_window,
            calibration_window=self.config.calibration_window
        )
        self.metacognitive_monitor = MetacognitiveMonitor(metacognitive_config)
        
        # Knowledge buffer for processing
        self.knowledge_buffer: List[Dict[str, Any]] = []
    
    def start_processing(self):
        """Start the consciousness processing loop in a separate thread."""
        if self.is_processing:
            return
            
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the consciousness processing loop."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def _processing_loop(self):
        """Main consciousness processing loop running at 100Hz."""
        while self.is_processing:
            # Update oscillator phases
            theta_phase, gamma_phase = self.oscillator.update_phases()
            
            # Process based on current state
            if self.processing_state == ProcessingState.DREAM_SYNTHESIS:
                # Check for theta-gamma coupling window
                if self.oscillator.get_coupling_window():
                    self._perform_dream_synthesis(gamma_phase)
            
            # Process knowledge buffer in all states except deep sleep
            if self.processing_state != ProcessingState.DEEP_SLEEP:
                self._process_knowledge_buffer()
            
            # Small delay to maintain ~100Hz processing rate
            time.sleep(0.01)
    
    def _perform_dream_synthesis(self, gamma_amplitude: float):
        """Perform dream synthesis during coupling windows."""
        if not self.knowledge_buffer:
            return
            
        # Convert knowledge buffer to fragments
        fragments = []
        for item in self.knowledge_buffer[:self.config.max_fragments_per_cycle]:
            item_fragments = self.memory_processor.fragment_knowledge(
                item['text'], item['embedding'])
            fragments.extend(item_fragments)
        
        if not fragments:
            return
            
        # Synthesize new concepts
        synthesized = self.dream_synthesizer.synthesize_concepts(fragments, gamma_amplitude)
        
        # Compete for consciousness access
        winners = self.workspace_manager.compete_for_consciousness(synthesized)
        
        # Update metacognitive monitor with synthesis quality
        if winners:
            avg_novelty = sum(content.novelty for content in winners) / len(winners)
            self.metacognitive_monitor.update_performance(avg_novelty, 0.8)  # High confidence in synthesis
    
    def _process_knowledge_buffer(self):
        """Process knowledge buffer items through memory fragmentation."""
        if not self.knowledge_buffer:
            return
            
        # Process a limited number of items per cycle
        items_to_process = min(10, len(self.knowledge_buffer))
        
        for i in range(items_to_process):
            if i >= len(self.knowledge_buffer):
                break
                
            item = self.knowledge_buffer[i]
            fragments = self.memory_processor.fragment_knowledge(
                item['text'], item['embedding'])
            
            # For now, we're just processing items - in a more complete implementation
            # we would store fragments in a memory system for later retrieval
            if fragments:
                # Simple performance metric based on fragment count
                performance = min(len(fragments) / 20.0, 1.0)  # Normalize to [0,1]
                self.metacognitive_monitor.update_performance(performance, 0.7)
        
        # Remove processed items
        self.knowledge_buffer = self.knowledge_buffer[items_to_process:]
    
    def add_knowledge(self, text: str, embedding: jnp.ndarray):
        """
        Add knowledge to the consciousness system for processing.
        
        Args:
            text: Knowledge text
            embedding: Knowledge embedding vector
        """
        knowledge_item = {
            'text': text,
            'embedding': embedding,
            'timestamp': time.time()
        }
        self.knowledge_buffer.append(knowledge_item)
    
    def set_processing_state(self, state: ProcessingState):
        """
        Set the consciousness processing state.
        
        Args:
            state: New processing state
        """
        self.processing_state = state
        
        # Clear workspace when entering deep sleep
        if state == ProcessingState.DEEP_SLEEP:
            self.workspace_manager.clear_workspace()
    
    def get_consciousness_status(self) -> Dict[str, Any]:
        """
        Get the current consciousness system status.
        
        Returns:
            Dictionary with consciousness system status
        """
        metacognitive_status = self.metacognitive_monitor.get_metacognitive_status()
        adjusted_thresholds = self.metacognitive_monitor.adjust_thresholds()
        
        return {
            'processing_state': self.processing_state.value,
            'is_processing': self.is_processing,
            'consciousness_level': metacognitive_status['self_awareness_level'],
            'calibration_error': metacognitive_status['calibration_error'],
            'knowledge_buffer_size': len(self.knowledge_buffer),
            'workspace': self.workspace_manager.get_workspace_status(),
            'adjusted_thresholds': adjusted_thresholds,
            'oscillator': {
                'theta_phase': self.oscillator.theta_phase,
                'gamma_phase': self.oscillator.gamma_phase,
                'in_coupling_window': self.oscillator.get_coupling_window()
            }
        }
