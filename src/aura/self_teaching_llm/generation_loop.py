#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import List, Tuple, Callable, Optional


def generate_text(prompt_embeddings: jnp.ndarray,
                  retrieval_core: Callable,
                  lang_core: Callable,
                  decoder: Callable,
                  max_len: int = 50,
                  temperature: float = 1.0) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """
    Generate text using the self-teaching LLM pipeline.
    
    Args:
        prompt_embeddings: Prompt embeddings [batch, embed_dim]
        retrieval_core: Function to retrieve context from memory
        lang_core: Function to process context with spiking dynamics
        decoder: Function to decode spiking rates to token probabilities
        max_len: Maximum generation length
        temperature: Sampling temperature for token generation
        
    Returns:
        Tuple of (generated_token_ids [batch, max_len], all_rates [List of [batch, hidden_dim]])
    """
    batch_size = prompt_embeddings.shape[0]
    
    # Initialize recurrent state for language core
    rnn_state = lang_core.initialize_state(batch_size)
    
    # Seed with prompt
    input_token = prompt_embeddings  # shape [batch, embed_dim]
    generated_tokens = []
    all_rates = []
    
    # Process prompt through retrieval core
    h_t = retrieval_core(input_token)  # [batch, hidden_dim]
    
    for t in range(max_len):
        # 1) Update spiking language core
        rate_out, rnn_state = lang_core(h_t, rnn_state)
        all_rates.append(rate_out)
        
        # 2) Decode to token probabilities
        token_probs = decoder(rate_out)  # [batch, vocab_size]
        
        # 3) Sample next token with temperature
        if temperature != 1.0:
            token_probs = token_probs ** (1.0 / temperature)
            token_probs = token_probs / jnp.sum(token_probs, axis=-1, keepdims=True)
        
        # Sample tokens
        key = jax.random.key(t)  # Deterministic key for reproducibility
        next_token_idx = jax.random.categorical(key, jnp.log(token_probs))  # [batch]
        generated_tokens.append(next_token_idx)
        
        # 4) Embed next token for subsequent retrieval
        # In a real implementation, this would be an actual embedding lookup
        # For now, we'll use a simple transformation
        input_token = jnp.expand_dims(next_token_idx.astype(jnp.float32), -1)  # [batch, 1]
        # Expand to match embedding dimension
        input_token = jnp.repeat(input_token, prompt_embeddings.shape[1], axis=-1)  # [batch, embed_dim]
        
        # 5) Retrieve context for next token
        h_t = retrieval_core(input_token)  # [batch, hidden_dim]
    
    # Stack generated tokens
    generated_token_ids = jnp.stack(generated_tokens, axis=1)  # [batch, max_len]
    
    return generated_token_ids, all_rates


def generate_text_with_consciousness(prompt_embeddings: jnp.ndarray,
                                   retrieval_core: Callable,
                                   lang_core: Callable,
                                   decoder: Callable,
                                   consciousness_system: Optional[Callable] = None,
                                   max_len: int = 50,
                                   temperature: float = 1.0) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """
    Generate text with consciousness system integration.
    
    Args:
        prompt_embeddings: Prompt embeddings [batch, embed_dim]
        retrieval_core: Function to retrieve context from memory
        lang_core: Function to process context with spiking dynamics
        decoder: Function to decode spiking rates to token probabilities
        consciousness_system: Consciousness system for context biasing (optional)
        max_len: Maximum generation length
        temperature: Sampling temperature for token generation
        
    Returns:
        Tuple of (generated_token_ids [batch, max_len], all_rates [List of [batch, hidden_dim]])
    """
    batch_size = prompt_embeddings.shape[0]
    
    # Get consciousness context if available
    conscious_context = None
    if consciousness_system is not None:
        workspace_status = consciousness_system.workspace_manager.get_workspace_status()
        conscious_context = workspace_status['contents']
    
    # Initialize recurrent state for language core
    rnn_state = lang_core.initialize_state(batch_size)
    
    # Seed with prompt
    input_token = prompt_embeddings  # shape [batch, embed_dim]
    generated_tokens = []
    all_rates = []
    
    # Process prompt through retrieval core
    h_t = retrieval_core(input_token)  # [batch, hidden_dim]
    
    for t in range(max_len):
        # Bias retrieval with conscious context if available
        # In a full implementation, this would modify the retrieval process
        # For now, we'll just pass through the same h_t
        
        # 1) Update spiking language core
        rate_out, rnn_state = lang_core(h_t, rnn_state)
        all_rates.append(rate_out)
        
        # 2) Decode to token probabilities
        token_probs = decoder(rate_out)  # [batch, vocab_size]
        
        # 3) Sample next token with temperature
        if temperature != 1.0:
            token_probs = token_probs ** (1.0 / temperature)
            token_probs = token_probs / jnp.sum(token_probs, axis=-1, keepdims=True)
        
        # Sample tokens
        key = jax.random.key(t)  # Deterministic key for reproducibility
        next_token_idx = jax.random.categorical(key, jnp.log(token_probs))  # [batch]
        generated_tokens.append(next_token_idx)
        
        # 4) Embed next token for subsequent retrieval
        # In a real implementation, this would be an actual embedding lookup
        # For now, we'll use a simple transformation
        input_token = jnp.expand_dims(next_token_idx.astype(jnp.float32), -1)  # [batch, 1]
        # Expand to match embedding dimension
        input_token = jnp.repeat(input_token, prompt_embeddings.shape[1], axis=-1)  # [batch, embed_dim]
        
        # 5) Retrieve context for next token
        h_t = retrieval_core(input_token)  # [batch, hidden_dim]
    
    # Stack generated tokens
    generated_token_ids = jnp.stack(generated_tokens, axis=1)  # [batch, max_len]
    
    return generated_token_ids, all_rates
