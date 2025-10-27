
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
fast_clean_pack.py — quick corpus normalize + exact dedup + paragraph packing
Optimized for speed on CPU. Near-dup off by default.
Usage:
  python fast_clean_pack.py --in corpus.txt --out corpus_clean.txt
"""
import argparse, sys, unicodedata, hashlib, re

TRANS = str.maketrans({"—":"-", "–":"-", "“":"\"", "”":"\"", "’":"'"})
RE_SPACE = re.compile(r"[ \t]+")

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(TRANS)
    s = RE_SPACE.sub(" ", s)
    return s.strip()

def paragraphs(stream):
    buf = []
    for line in stream:
        if line.strip() == "":
            if buf:
                yield "\n".join(buf).strip()
                buf = []
        else:
            buf.append(line.rstrip("\n"))
    if buf:
        yield "\n".join(buf).strip()

def write_clean(in_path: str, out_path: str):
    seen = set()
    with open(out_path, "w", encoding="utf-8") as out:
        first = True
        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            for para in paragraphs(f):
                p = norm(para)
                if not p: continue
                h = hashlib.blake2b(p.encode("utf-8"), digest_size=8).hexdigest()
                if h in seen: 
                    continue
                seen.add(h)
                if not first:
                    out.write("\n\n")
                out.write(p)
                first = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    write_clean(args.inp, args.out)

if __name__ == "__main__":
    main()
