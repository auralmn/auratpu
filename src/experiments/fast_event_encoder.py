#!/usr/bin/env python3
"""
Fast EventPatternEncoder that loads preprocessed keyword tensors from .pt file
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import re

class FastEventPatternEncoder(nn.Module):
    """
    Optimized event pattern encoder using preprocessed keyword tensors
    Loads from .pt file for instant initialization
    """
    
    def __init__(self, pt_file_path: str, device: Optional[torch.device] = None):
        super().__init__()
        
        self.device = device or torch.device('cpu')
        
        # Load preprocessed data
        print(f"Loading preprocessed keywords from {pt_file_path}...")
        data = torch.load(pt_file_path, map_location=self.device)
        
        # Store core data
        self.d_model = data['metadata']['d_model']
        self.event_names = data['event_names']
        self.keyword_list = data['keyword_list']
        self.keyword_indices = data['keyword_indices']
        
        # Register preprocessed tensors as buffers (non-trainable)
        self.register_buffer('event_patterns', data['event_patterns'])  # [num_events, d_model]
        self.register_buffer('scoring_matrix', data['scoring_matrix'])  # [num_keywords, num_events]
        
        # Learnable adaptation weights
        num_events = len(self.event_names)
        self.event_weights = nn.Parameter(torch.ones(num_events))
        self.event_biases = nn.Parameter(torch.zeros(num_events))
        self.global_scale = nn.Parameter(torch.tensor(1.0))
        
        # Temporal patterns if available
        self.temporal_patterns = data.get('temporal_patterns', {})
        
        # Compile regex for faster keyword matching
        self._compile_keyword_regex()
        
        print(f"Loaded {len(self.keyword_list)} keywords for {num_events} event types")
        print(f"Pattern dimensions: {self.d_model}")
    
    def _compile_keyword_regex(self):
        """Compile regex patterns for ultra-fast keyword matching"""
        # Sort keywords by length (longest first) for better matching
        sorted_keywords = sorted(self.keyword_list, key=len, reverse=True)
        
        # Escape special regex characters and create pattern
        escaped_keywords = [re.escape(kw) for kw in sorted_keywords]
        
        # Create word boundary pattern for exact matches
        pattern = r'\b(' + '|'.join(escaped_keywords) + r')\b'
        self.keyword_regex = re.compile(pattern, re.IGNORECASE)
        
        # Create mapping from matched keyword to indices
        self.matched_keyword_to_idx = {kw: self.keyword_indices[kw] for kw in self.keyword_list}
    
    def fast_keyword_extraction(self, text: str) -> torch.Tensor:
        """
        Ultra-fast keyword extraction using compiled regex
        Returns: event_scores tensor [num_events]
        """
        if not text:
            return torch.zeros(len(self.event_names), device=self.device)
        
        # Find all keyword matches
        matches = self.keyword_regex.findall(text.lower())
        
        if not matches:
            return torch.zeros(len(self.event_names), device=self.device)
        
        # Count keyword frequencies
        keyword_counts = {}
        for match in matches:
            keyword_counts[match] = keyword_counts.get(match, 0) + 1
        
        # Create keyword activation vector
        keyword_activations = torch.zeros(len(self.keyword_list), device=self.device)
        
        for keyword, count in keyword_counts.items():
            if keyword in self.matched_keyword_to_idx:
                idx = self.matched_keyword_to_idx[keyword]
                keyword_activations[idx] = float(count)
        
        # Matrix multiply to get event scores: [num_keywords] @ [num_keywords, num_events] -> [num_events]
        event_scores = torch.matmul(keyword_activations, self.scoring_matrix)
        
        return event_scores
    
    def encode_text_to_patterns(self, text: str) -> torch.Tensor:
        """
        Convert text to spike patterns using fast keyword matching
        Returns: composite_pattern [1, d_model]
        """
        # Fast keyword-based event scoring
        event_scores = self.fast_keyword_extraction(text)
        
        # Apply learnable weights and biases
        adapted_scores = event_scores * self.event_weights + self.event_biases
        adapted_scores = torch.relu(adapted_scores)  # Ensure non-negative
        
        # Normalize scores
        total_score = torch.sum(adapted_scores)
        if total_score > 0:
            normalized_scores = adapted_scores / total_score
        else:
            # No keywords found, return zero pattern
            return torch.zeros(1, self.d_model, device=self.device)
        
        # Weighted combination of event patterns
        # [num_events] @ [num_events, d_model] -> [d_model]
        composite_pattern = torch.matmul(normalized_scores, self.event_patterns)
        
        # Apply global scaling
        composite_pattern = composite_pattern * self.global_scale
        
        return composite_pattern.unsqueeze(0)  # Add batch dimension
    
    def get_event_analysis(self, text: str) -> Dict[str, Any]:
        """
        Detailed analysis of detected events for debugging/visualization
        """
        event_scores = self.fast_keyword_extraction(text)
        adapted_scores = event_scores * self.event_weights + self.event_biases
        adapted_scores = torch.relu(adapted_scores)
        
        # Get top events
        top_scores, top_indices = torch.topk(adapted_scores, k=min(5, len(self.event_names)))
        
        analysis = {
            'detected_events': [],
            'total_keywords_found': 0,
            'pattern_intensity': 0.0
        }
        
        for score, idx in zip(top_scores, top_indices):
            if score > 0:
                event_name = self.event_names[idx]
                analysis['detected_events'].append({
                    'event_type': event_name,
                    'score': float(score),
                    'normalized_score': float(score / torch.sum(adapted_scores)) if torch.sum(adapted_scores) > 0 else 0.0
                })
        
        # Count actual keyword matches
        matches = self.keyword_regex.findall(text.lower()) if text else []
        analysis['total_keywords_found'] = len(matches)
        analysis['unique_keywords_found'] = len(set(matches))
        analysis['keyword_matches'] = list(set(matches))
        
        # Pattern intensity
        if torch.sum(adapted_scores) > 0:
            pattern = self.encode_text_to_patterns(text)
            analysis['pattern_intensity'] = float(torch.sum(torch.abs(pattern)))
        
        return analysis
    
    def to(self, device):
        """Override to method for device management"""
        super().to(device)
        self.device = device
        return self
