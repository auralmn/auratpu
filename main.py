import argparse, sys, os, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "experiments"))

def run_tests(k=None, start_dir="tests", verbosity=1):
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=start_dir, pattern="test*.py")
    if k:
        def _filter(s, pat):
            new = unittest.TestSuite()
            for t in s:
                if isinstance(t, unittest.TestSuite):
                    sub = _filter(t, pat)
                    if sub.countTestCases():
                        new.addTest(sub)
                else:
                    name = t.id()
                    if pat in name:
                        new.addTest(t)
            return new
        suite = _filter(suite, k)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    res = runner.run(suite)
    return 0 if res.wasSuccessful() else 1

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    p_test = sp.add_parser("test")
    p_test.add_argument("-k", "--keyword", default=None)
    p_test.add_argument("-v", "--verbose", action="count", default=0)

    p_tm = sp.add_parser("train-mini")
    p_tm.add_argument("args", nargs=argparse.REMAINDER)

    p_tms = sp.add_parser("train-mini-small")
    p_tms.add_argument("args", nargs=argparse.REMAINDER)

    p_pre = sp.add_parser("pretrain-snn")
    p_pre.add_argument("args", nargs=argparse.REMAINDER)

    p_cp = sp.add_parser("clean-pack")
    p_cp.add_argument("--in", dest="inp", required=True)
    p_cp.add_argument("--out", dest="out", required=True)
    p_cp.add_argument("--max-lines", type=int, default=None)

    p_fcp = sp.add_parser("fast-clean-pack")
    p_fcp.add_argument("--in", dest="inp", required=True)
    p_fcp.add_argument("--out", dest="out", required=True)

    p_ptb = sp.add_parser("pack-to-bin")
    p_ptb.add_argument("args", nargs=argparse.REMAINDER)

    p_mem = sp.add_parser("memory-ingest")
    p_mem.add_argument("args", nargs=argparse.REMAINDER)

    p_jax = sp.add_parser("train-jax")
    p_jax.add_argument("args", nargs=argparse.REMAINDER)

    p_jaxv = sp.add_parser("validate-jax")

    p_comp = sp.add_parser("intent-compass")
    p_comp.add_argument("args", nargs=argparse.REMAINDER)

    p_aoc = sp.add_parser("ao-classify")
    p_aoc.add_argument("args", nargs=argparse.REMAINDER)

    p_aocj = sp.add_parser("ao-classify-jax")
    p_aocj.add_argument("args", nargs=argparse.REMAINDER)

    args = ap.parse_args()

    if args.cmd == "test":
        verbosity = 1 + int(args.verbose)
        code = run_tests(k=args.keyword, start_dir="tests", verbosity=verbosity)
        sys.exit(code)
    elif args.cmd == "train-mini":
        import aura_mini_train_tpu as mod
        argv = ["prog"] + args.args
        save = sys.argv; sys.argv = argv
        try:
            mod.main()
        finally:
            sys.argv = save
    elif args.cmd == "train-mini-small":
        import aura_mini_train_tpu_small as mod
        argv = ["prog"] + args.args
        save = sys.argv; sys.argv = argv
        try:
            mod.main()
        finally:
            sys.argv = save
    elif args.cmd == "pretrain-snn":
        import snn_llm_pretrainer_tpu as mod
        argv = ["prog"] + args.args
        save = sys.argv; sys.argv = argv
        try:
            mod.main()
        finally:
            sys.argv = save
    elif args.cmd == "clean-pack":
        from clean_dedup_pack import process as cdp_process
        cdp_process(args.inp, args.out, max_lines=args.max_lines)
    elif args.cmd == "fast-clean-pack":
        from fast_clean_pack import write_clean as f_write
        f_write(args.inp, args.out)
    elif args.cmd == "pack-to-bin":
        import pack_to_bin as mod
        argv = ["prog"] + args.args
        save = sys.argv; sys.argv = argv
        try:
            mod.main()
        finally:
            sys.argv = save
    elif args.cmd == "memory-ingest":
        import aura_memory_ingest as mod
        argv = ["prog"] + args.args
        save = sys.argv; sys.argv = argv
        try:
            mod.main()
        finally:
            sys.argv = save
    elif args.cmd == "train-jax":
        import aura_mini_train_jax as mod
        argv = ["prog"] + args.args
        save = sys.argv; sys.argv = argv
        try:
            mod.main()
        finally:
            sys.argv = save
    elif args.cmd == "validate-jax":
        from neuromorphic_srwkv_jax import NeuromorphicSRWKVJax
        import jax
        model = NeuromorphicSRWKVJax(embedding_dim=64, num_heads=4, attn_mode='streaming', block_size_q=8, block_size_kv=8)
        key = jax.random.key(0)
        from neuromorphic_srwkv_jax import validate_model
        res = validate_model(model, key)
        print(res)
    elif args.cmd == "intent-compass":
        import shadowbank_compass as mod
        argv = ["prog"] + args.args
        save = sys.argv; sys.argv = argv
        try:
            mod.main()
        finally:
            sys.argv = save
    elif args.cmd == "ao-classify":
        import addition_only_classifier_torch as mod
        argv = ["prog"] + args.args
        save = sys.argv; sys.argv = argv
        try:
            mod.main()
        finally:
            sys.argv = save
    elif args.cmd == "ao-classify-jax":
        import addition_only_classifier_jax as mod
        argv = ["prog"] + args.args
        save = sys.argv; sys.argv = argv
        try:
            mod.main()
        finally:
            sys.argv = save

if __name__ == "__main__":
    main()
