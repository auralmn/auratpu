
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
AURA ShadowBank (Minimal): persistent memory + embeddings + URL ingest
- Storage: JSONL metadata + torch tensor embeddings (cosine)
- Embeddings: 
    * default = byte-histogram (259-dim) -> random projection to d_embed (e.g., 512)
    * custom embedder callable supported
- Retrieval: cosine TOP-K
- Ingest from URL: stdlib urllib + basic HTML stripping (no external deps)
"""

from __future__ import annotations
import os, io, json, time, random, math, re, hashlib
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Callable, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from html.parser import HTMLParser

import torch

@dataclass
class MemoryItem:
    id: str
    url: Optional[str]
    title: Optional[str]
    text: str
    ts: float
    tags: List[str]
    valence: float
    meta: Dict[str, Any]

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.data: List[str] = []
    def handle_data(self, d): self.data.append(d)
    def get_text(self): return " ".join(self.data)

def _strip_html(html: str) -> str:
    hp = _HTMLStripper()
    hp.feed(html)
    return hp.get_text()

def _fetch_url(url: str, timeout: int = 15) -> Tuple[str, str]:
    # Returns (title, text)
    ua = "AURA-Memory/0.1 (+https://example.invalid/)"
    req = Request(url, headers={"User-Agent": ua})
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or 'utf-8'
        raw = resp.read().decode(charset, errors='ignore')
    # naive title extraction
    title = ""
    m = re.search(r"<title>(.*?)</title>", raw, flags=re.IGNORECASE | re.DOTALL)
    if m: title = re.sub(r"\s+", " ", m.group(1)).strip()
    text = _strip_html(raw)
    text = re.sub(r"\s+", " ", text).strip()
    return title, text

class RandomProjectionEmbedder:
    """Byte-histogram -> RP to d_embed (device-agnostic)."""
    def __init__(self, d_out: int = 512, seed: int = 0):
        self.d_in = 259  # byte vocab + PAD/BOS/EOS used by tokenizer
        self.d_out = d_out  # <-- assign first (NO walrus)
        g = torch.Generator().manual_seed(seed)
        # Achlioptas RP (sparse-ish): entries in {-1,0,1}
        probs = torch.rand((self.d_out, self.d_in), generator=g)
        mat = torch.zeros_like(probs)
        mat[probs < 1/6] = -1.0
        mat[(probs >= 1/6) & (probs < 2/6)] = 1.0
        self.R = mat / math.sqrt(self.d_in)

    def __call__(self, ids: List[int]) -> torch.Tensor:
        # ids are in byte space with special markers; build histogram over 0..258
        hist = torch.zeros(self.d_in, dtype=torch.float32)
        if ids:
            # clamp to valid range
            t = torch.tensor(ids, dtype=torch.long)
            t = torch.clamp(t, 0, self.d_in - 1)
            # bincount to histogram length d_in
            bc = torch.bincount(t, minlength=self.d_in).to(torch.float32)
            hist = bc / (float(bc.sum().item()) + 1e-9)
        # Project: [d_out, d_in] @ [d_in] -> [d_out]
        return self.R @ hist

class MemoryStore:
    def __init__(self, path: str, d_embed: int = 512, embedder: Optional[Callable[[List[int]], torch.Tensor]] = None):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.meta_path = os.path.join(path, "store.jsonl")
        self.emb_path = os.path.join(path, "embeddings.pt")
        self.items: List[MemoryItem] = []
        self.embeddings = torch.empty(0, d_embed)
        self.embedder = embedder if embedder is not None else RandomProjectionEmbedder(d_embed)
        if os.path.exists(self.meta_path):
            self._load()

    def _load(self):
        self.items = []
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.items.append(MemoryItem(**json.loads(line)))
        if os.path.exists(self.emb_path):
            self.embeddings = torch.load(self.emb_path, map_location='cpu')
        else:
            self.embeddings = torch.zeros((len(self.items), self.embedder.R.shape[0]))

    def _save_append(self, item: MemoryItem, emb: torch.Tensor):
        with open(self.meta_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        if self.embeddings.numel() == 0:
            self.embeddings = emb.unsqueeze(0)
        else:
            self.embeddings = torch.cat([self.embeddings, emb.unsqueeze(0)], dim=0)
        torch.save(self.embeddings, self.emb_path)

    def add_text(self, text: str, *, url: Optional[str] = None, title: Optional[str] = None,
                 tags: Optional[List[str]] = None, valence: Optional[float] = None, meta: Optional[Dict[str, Any]] = None) -> str:
        # id = stable hash of (url|text)
        h = hashlib.sha1((url or "") .encode() + b"|" + text[:1000].encode('utf-8', errors='ignore')).hexdigest()
        item = MemoryItem(
            id=h, url=url, title=title, text=text, ts=time.time(),
            tags=tags or [], valence=float(valence if valence is not None else self._heuristic_valence(text)),
            meta=meta or {}
        )
        emb = self.embedder(self._bytes_to_ids(text))
        self.items.append(item)
        self._save_append(item, emb)
        return h

    def ingest_url(self, url: str, *, min_chars: int = 200) -> Optional[str]:
        try:
            title, text = _fetch_url(url)
        except Exception as e:
            print(f"[ingest_url] failed: {url} | {e}")
            return None
        if len(text) < min_chars:
            print(f"[ingest_url] too short: {url} ({len(text)} chars)")
            return None
        return self.add_text(text, url=url, title=title, tags=['web'])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        if len(self.items) == 0:
            return []
        q_emb = self.embedder(self._bytes_to_ids(query)).unsqueeze(0)  # [1,d]
        E = self.embeddings  # [N,d]
        # cosine sim
        qn = q_emb / (q_emb.norm(dim=1, keepdim=True) + 1e-9)
        En = E / (E.norm(dim=1, keepdim=True) + 1e-9)
        sims = (qn @ En.t()).squeeze(0)  # [N]
        vals, idx = torch.topk(sims, k=min(top_k, sims.numel()))
        out = []
        for s, i in zip(vals.tolist(), idx.tolist()):
            out.append((self.items[i], float(s)))
        return out

    @staticmethod
    def _heuristic_valence(text: str) -> float:
        exclam = text.count('!')
        caps = sum(1 for c in text if c.isupper())
        length = max(len(text), 1)
        return min(1.0, 0.1*exclam + 0.5*(caps/length))

    @staticmethod
    def _bytes_to_ids(text: str) -> List[int]:
        b = list(text.encode('utf-8', errors='ignore'))
        return [0] + b + [1]  # BOS/EOS placeholders in 0/1 positions
