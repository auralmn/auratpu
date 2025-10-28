#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import Dict, List, Optional, Any

import jax
import jax.numpy as jnp

try:
    import torch  # used only for loading .pt preprocessed file
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


class FastEventPatternEncoderJax:
    """
    JAX version of FastEventPatternEncoder.
    Loads preprocessed tensors from a .pt file (via torch), converts to jnp.
    """
    def __init__(self, pt_file_path: str):
        if not HAS_TORCH:
            raise ImportError("torch is required to load the preprocessed .pt file for JAX encoder")

        data = torch.load(pt_file_path, map_location='cpu')

        self.d_model: int = int(data['metadata']['d_model'])
        self.event_names: List[str] = list(data['event_names'])
        self.keyword_list: List[str] = list(data['keyword_list'])
        self.keyword_indices: Dict[str, int] = dict(data['keyword_indices'])

        # Convert tensors to jnp
        def to_jnp(t):
            if isinstance(t, torch.Tensor):
                return jnp.asarray(t.cpu().numpy())
            return jnp.asarray(t)

        self.event_patterns = to_jnp(data['event_patterns'])  # [num_events, d_model]
        self.scoring_matrix = to_jnp(data['scoring_matrix'])  # [num_keywords, num_events]

        # Trainable equivalents as arrays (non-trainable here)
        num_events = len(self.event_names)
        self.event_weights = jnp.ones((num_events,), dtype=jnp.float32)
        self.event_biases = jnp.zeros((num_events,), dtype=jnp.float32)
        self.global_scale = jnp.asarray(1.0, dtype=jnp.float32)

        # Compile regex
        self._compile_keyword_regex()

    def _compile_keyword_regex(self):
        sorted_keywords = sorted(self.keyword_list, key=len, reverse=True)
        escaped = [re.escape(kw) for kw in sorted_keywords]
        pattern = r'\b(' + '|'.join(escaped) + r')\b'
        self.keyword_regex = re.compile(pattern, re.IGNORECASE)
        self.matched_keyword_to_idx = {kw: self.keyword_indices[kw] for kw in self.keyword_list}

    def fast_keyword_extraction(self, text: str) -> jnp.ndarray:
        if not text:
            return jnp.zeros((len(self.event_names),), dtype=jnp.float32)
        matches = self.keyword_regex.findall(text.lower())
        if not matches:
            return jnp.zeros((len(self.event_names),), dtype=jnp.float32)

        # Build keyword activation vector
        activ = jnp.zeros((len(self.keyword_list),), dtype=jnp.float32)
        # Count in Python for simplicity
        counts: Dict[str, int] = {}
        for m in matches:
            counts[m] = counts.get(m, 0) + 1
        for kw, c in counts.items():
            idx = self.matched_keyword_to_idx.get(kw, None)
            if idx is not None:
                activ = activ.at[idx].set(float(c))

        # event_scores = activ @ scoring_matrix  -> [num_events]
        scores = activ @ self.scoring_matrix
        return scores.astype(jnp.float32)

    def encode_text_to_patterns(self, text: str) -> jnp.ndarray:
        event_scores = self.fast_keyword_extraction(text)
        adapted = jnp.maximum(0.0, event_scores * self.event_weights + self.event_biases)
        total = jnp.sum(adapted)
        def _zero():
            return jnp.zeros((1, self.d_model), dtype=jnp.float32)
        def _nonzero():
            norm = adapted / total
            comp = norm @ self.event_patterns  # [d_model]
            comp = comp * self.global_scale
            return comp[None, :]
        return jax.lax.cond(total > 0, _nonzero, _zero)

    def get_event_analysis(self, text: str) -> Dict[str, Any]:
        event_scores = self.fast_keyword_extraction(text)
        adapted = jnp.maximum(0.0, event_scores * self.event_weights + self.event_biases)
        k = min(5, len(self.event_names))
        # Top-k using jnp; fallback: sort
        idx = jnp.argsort(-adapted)[:k]
        top_scores = adapted[idx]
        total = float(jnp.sum(adapted))
        analysis = {
            'detected_events': [],
            'total_keywords_found': 0,
            'pattern_intensity': 0.0,
        }
        for s, i in zip(list(map(float, top_scores)), list(map(int, idx))):
            if s > 0:
                analysis['detected_events'].append({
                    'event_type': self.event_names[i],
                    'score': s,
                    'normalized_score': (s / total) if total > 0 else 0.0,
                })
        matches = self.keyword_regex.findall(text.lower()) if text else []
        analysis['total_keywords_found'] = len(matches)
        analysis['unique_keywords_found'] = len(set(matches))
        analysis['keyword_matches'] = list(set(matches))
        if total > 0:
            patt = self.encode_text_to_patterns(text)
            analysis['pattern_intensity'] = float(jnp.sum(jnp.abs(patt)))
        return analysis
