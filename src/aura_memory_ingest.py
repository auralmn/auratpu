
#!/usr/bin/env python3
import argparse, sys, json
from memory_store import MemoryStore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memory-dir', type=str, required=True)
    ap.add_argument('--url', type=str, default=None, help='Single URL to ingest')
    ap.add_argument('--urls-file', type=str, default=None, help='File with URLs (one per line)')
    args = ap.parse_args()
    ms = MemoryStore(args.memory_dir)
    added = []
    if args.url:
        hid = ms.ingest_url(args.url)
        if hid: added.append(hid)
    if args.urls_file:
        with open(args.urls_file, 'r') as f:
            for line in f:
                u = line.strip()
                if not u: continue
                hid = ms.ingest_url(u)
                if hid: added.append(hid)
    print(json.dumps({'added': added, 'total_items': len(ms.items)}))

if __name__ == "__main__":
    main()
