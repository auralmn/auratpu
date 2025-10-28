# clean_dedup_pack.py
import re, sys, unicodedata, hashlib, random
from collections import defaultdict

def norm(txt:str)->str:
    t = unicodedata.normalize("NFKC", txt)
    t = t.replace("—","-").replace("–","-").replace("“","\"").replace("”","\"").replace("’","'")
    t = re.sub(r"[ \t]+"," ", t)
    t = re.sub(r"\n{3,}","\n\n", t)
    return t.strip()

def simhash(tokens, bits=64):
    import mmh3
    v=[0]*bits
    for tok in tokens:
        h = mmh3.hash(tok, signed=False)
        for b in range(bits):
            v[b] += 1 if (h>>b)&1 else -1
    out=0
    for b in range(bits):
        if v[b]>=0: out |= (1<<b)
    return out

def neardup_key(paragraph):
    toks = re.findall(r"[a-z0-9]+", paragraph.lower())
    # 5-gram shingles for local near-dup
    shingles = [" ".join(toks[i:i+5]) for i in range(max(1,len(toks)-4))]
    return simhash(shingles)

def process(in_path, out_path, max_lines=None):
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read() if max_lines is None else "".join([next(f) for _ in range(max_lines)])
    raw = norm(raw)

    # split by blank lines => paragraphs
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]

    # exact dedup
    seen_hash = set()
    uniq = []
    for p in paras:
        h = hashlib.sha1(p.encode("utf-8")).hexdigest()
        if h not in seen_hash:
            seen_hash.add(h)
            uniq.append(p)

    # light near-dup within this corpus (banded simhash buckets)
    buckets = defaultdict(list)
    kept = []
    for p in uniq:
        key = neardup_key(p)
        band = (key >> 48)  # top 16 bits as bucket
        dup = False
        for k2 in buckets[band]:
            # Hamming <= 3 -> drop
            if (bin(key ^ k2).count("1") <= 3):
                dup = True; break
        if not dup:
            buckets[band].append(key)
            kept.append(p)

    with open(out_path, "w", encoding="utf-8") as w:
        w.write("\n\n".join(kept) + "\n")

if __name__ == "__main__":
    process(sys.argv[1], sys.argv[2])
