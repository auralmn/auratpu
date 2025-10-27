#!/usr/bin/env bash

PYTHONUNBUFFERED=1 python -u aura_mini_train_tpu.py --tpu \
  --bin-dir data_bin --epochs 1 --batch-size 8 --microbatches 4 \
  --seq-len 2048 --dim 512 --heads 8 --layers 6 \
  --attn-mode streaming --block-q 128 --block-kv 256 --lr 2e-4 \
  --ema-beta 0.98 --avg-window 100 --log-every 1