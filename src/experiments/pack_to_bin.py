
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
pack_to_bin.py — byte-tokenize and pack to fixed windows; writes tokens.bin (uint16) and idx.npy
Usage:
  python pack_to_bin.py --in corpus_clean.txt --out-dir data_bin --seq-len 2048
"""
import argparse, os, numpy as np

PAD, BOS, EOS, MEM, SEP = 256, 257, 258, 259, 260

def encode_bytes(s: str, add_special=True):
    ids = list(s.encode("utf-8", errors="ignore"))
    return ([BOS] + ids + [EOS]) if add_special else ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    toks = []
    with open(args.inp, "r", encoding="utf-8") as f:
        buf = []
        for line in f:
            if line.strip()=="" and buf:
                s = "\n".join(buf).strip()
                toks.extend(encode_bytes(s, add_special=True))
                buf = []
            else:
                buf.append(line.rstrip("\n"))
        if buf:
            s = "\n".join(buf).strip()
            toks.extend(encode_bytes(s, add_special=True))

    T = args.seq_len + 1
    if len(toks) % T != 0:
        rem = T - (len(toks) % T)
        toks.extend([PAD]*rem)

    import numpy as np
    arr = np.asarray(toks, dtype=np.uint16)
    n_samples = len(arr) // T
    idx = np.arange(0, len(arr), T, dtype=np.int64)

    arr.tofile(os.path.join(args.out_dir, "tokens.bin"))
    np.save(os.path.join(args.out_dir, "idx.npy"), idx)
    meta = {"seq_len": args.seq_len, "vocab_size": 261, "n_samples": int(n_samples)}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as w:
        import json; json.dump(meta, w, indent=2)
    print(meta)

if __name__ == "__main__":
    main()
