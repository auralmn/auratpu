#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import io
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

import math
import numpy as np

from memory_store import MemoryStore


@dataclass
class ShadowRecord:
    memory_id: str
    intent: List[float]
    valence: float
    tags: List[str]
    ts: float


class ShadowBank:
    def __init__(self, memory_dir: str, intent_dim: int = 512):
        self.ms = MemoryStore(memory_dir, d_embed=intent_dim)
        self.path = os.path.join(memory_dir, 'shadowbank.jsonl')
        self.records: List[ShadowRecord] = []
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.records.append(ShadowRecord(**json.loads(line)))

    def _save_append(self, rec: ShadowRecord):
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(rec)) + '\n')

    def embed_intent(self, text: str) -> np.ndarray:
        ids = self.ms._bytes_to_ids(text)
        v = self.ms.embedder(ids).detach().cpu().numpy().astype(np.float32)
        return v

    def add_text(self, text: str, *, intent_text: Optional[str] = None,
                 valence: Optional[float] = None, tags: Optional[List[str]] = None, ts: Optional[float] = None) -> str:
        hid = self.ms.add_text(text, url=None, title=None, tags=tags or [], valence=valence, meta={})
        itxt = intent_text if intent_text is not None else text
        ivec = self.embed_intent(itxt)
        rec = ShadowRecord(memory_id=hid,
                           intent=ivec.tolist(),
                           valence=float(valence if valence is not None else 0.0),
                           tags=tags or [],
                           ts=float(ts if ts is not None else time.time()))
        self.records.append(rec)
        self._save_append(rec)
        return hid


class IntentCompass:
    def __init__(self, bank: ShadowBank):
        self.bank = bank

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        an = a / (np.linalg.norm(a) + 1e-9)
        bn = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(an, bn))

    def retrieve(self, query: str, top_k: int = 5, lambda_decay: float = 0.0,
                 use_valence: bool = False, valence_weight: float = 0.0,
                 now: Optional[float] = None) -> List[Tuple[ShadowRecord, float]]:
        q = self.bank.embed_intent(query)
        now_t = float(now if now is not None else time.time())
        out: List[Tuple[ShadowRecord, float]] = []
        for rec in self.bank.records:
            ivec = np.asarray(rec.intent, dtype=np.float32)
            s = self._cosine(q, ivec)
            if lambda_decay > 0:
                age = max(0.0, now_t - float(rec.ts))
                s *= math.exp(-lambda_decay * age)
            if use_valence and valence_weight != 0.0:
                s *= (1.0 + valence_weight * float(rec.valence))
            out.append((rec, float(s)))
        out.sort(key=lambda p: p[1], reverse=True)
        return out[:max(0, min(top_k, len(out)))]


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)

    p_add = sp.add_parser('add')
    p_add.add_argument('--memory-dir', required=True)
    p_add.add_argument('--text', required=True)
    p_add.add_argument('--intent', default=None)
    p_add.add_argument('--valence', type=float, default=0.0)
    p_add.add_argument('--tags', default='', help='comma-separated')

    p_q = sp.add_parser('query')
    p_q.add_argument('--memory-dir', required=True)
    p_q.add_argument('--intent', required=True)
    p_q.add_argument('--top-k', type=int, default=5)
    p_q.add_argument('--lambda', dest='lmb', type=float, default=0.0)
    p_q.add_argument('--valence-weight', type=float, default=0.0)

    args = ap.parse_args()

    if args.cmd == 'add':
        bank = ShadowBank(args.memory_dir)
        tags = [t.strip() for t in args.tags.split(',') if t.strip()]
        hid = bank.add_text(args.text, intent_text=args.intent, valence=args.valence, tags=tags)
        print(json.dumps({'id': hid, 'total_records': len(bank.records)}))
    elif args.cmd == 'query':
        bank = ShadowBank(args.memory_dir)
        comp = IntentCompass(bank)
        results = comp.retrieve(args.intent, top_k=args.top_k, lambda_decay=args.lmb, use_valence=(args.valence_weight != 0.0), valence_weight=args.valence_weight)
        out = []
        for rec, s in results:
            out.append({'memory_id': rec.memory_id, 'score': s, 'valence': rec.valence, 'tags': rec.tags, 'ts': rec.ts})
        print(json.dumps({'results': out}))


if __name__ == '__main__':
    main()
