#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import List, Optional, Tuple, Callable

from .spiking_retrieval_core import SpikingRetrievalCore
from .spiking_language_core import SpikingLanguageCore
from .token_decoder import TokenDecoder
from .generation_loop import generate_text_with_consciousness


class SelfTeachingAdapter:
    """
    Self-teaching adapter for LLM that integrates consciousness context.
    Coordinates between spiking retrieval, language core, and token decoding.
    """
    
    def __init__(self, 
                 embed_dim: int = 768,
                 hidden_dim: int = 512,
                 vocab_size: int = 32000,
                 num_experts: int = 16):
        """
        Initialize self-teaching adapter.
        
        Args:
            embed_dim: Embedding dimension
            hidden_dim: Hidden state dimension for spiking cores
            vocab_size: Vocabulary size for token decoding
            num_experts: Number of experts in Liquid-MoE retrieval core
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        
        # Initialize components
        self.retrieval_core = SpikingRetrievalCore(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            expert_dim=embed_dim
        )
        
        self.lang_core = SpikingLanguageCore(
            hidden_dim=hidden_dim
        )
        
        self.token_decoder = TokenDecoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size
        )
        
        # Component parameters (will be initialized during first use)
        self.retrieval_params = None
        self.lang_params = None
        self.decoder_params = None
        
        # Random key for initialization
        self.key = jax.random.key(0)
    
    def initialize_parameters(self, batch_size: int = 1):
        """
        Initialize all component parameters.
        
        Args:
            batch_size: Batch size for parameter initialization
        """
        # Create dummy inputs for initialization
        dummy_embeddings = jnp.ones((batch_size, self.embed_dim))
        dummy_rates = jnp.ones((batch_size, self.hidden_dim))
        
        # Initialize parameters
        self.retrieval_params = self.retrieval_core.init(self.key, dummy_embeddings)
        self.lang_params = self.lang_core.init(self.key, dummy_rates, 
                                             self.lang_core.initialize_state(batch_size))
        self.decoder_params = self.token_decoder.init(self.key, dummy_rates)
    
    def generate_with_consciousness(self,
                                  prompt_embeddings: jnp.ndarray,
                                  consciousness_system: Optional[Callable] = None,
                                  max_len: int = 50,
                                  temperature: float = 1.0) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Generate text with consciousness system integration.
        
        Args:
            prompt_embeddings: Prompt embeddings [batch, embed_dim]
            consciousness_system: Consciousness system for context biasing (optional)
            max_len: Maximum generation length
            temperature: Sampling temperature for token generation
            
        Returns:
            Tuple of (generated_token_ids [batch, max_len], all_rates [List of [batch, hidden_dim]])
        """
        # Initialize parameters if not already done
        if self.retrieval_params is None:
            self.initialize_parameters(prompt_embeddings.shape[0])
        
        # Create component functions with parameters applied
        def retrieval_fn(query_embedding):
            return self.retrieval_core.apply(self.retrieval_params, 
                                          query_embedding)
        
        # Add the retrieve_context method to the function
        retrieval_fn.retrieve_context = retrieval_fn
        
        def lang_fn(input_state, prev_state):
            return self.lang_core.apply(self.lang_params, input_state, prev_state)
        
        def decoder_fn(rate_vector):
            return self.token_decoder.apply(self.decoder_params, rate_vector)
        
        # Add required methods to functions
        lang_fn.initialize_state = lambda batch_size: self.lang_core.initialize_state(batch_size)
        
        # Generate text with consciousness
        generated_tokens, all_rates = generate_text_with_consciousness(
            prompt_embeddings,
            retrieval_fn,
            lang_fn,
            decoder_fn,
            consciousness_system=consciousness_system,
            max_len=max_len,
            temperature=temperature
        )
        
        return generated_tokens, all_rates
    
    def teach_self(self, generated_text: str, generated_embeddings: jnp.ndarray):
        """
        Teach the model from its own generated text.
        
        Args:
            generated_text: Text generated by the model
            generated_embeddings: Embeddings of the generated text
        """
        # In a full implementation, this would:
        # 1. Add generated text to memory fragments
        # 2. Update consciousness system with new knowledge
        # 3. Potentially update component parameters through learning
        pass
    
    def get_component_status(self) -> dict:
        """
        Get status of all self-teaching components.
        
        Returns:
            Dictionary with component status information
        """
        return {
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'vocab_size': self.vocab_size,
            'num_experts': self.num_experts,
            'parameters_initialized': self.retrieval_params is not None
        }
