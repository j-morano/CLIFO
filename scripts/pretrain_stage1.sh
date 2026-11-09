#!/usr/bin/env bash
# Stage 1: B-scan encoder (masked B-scan modelling + layer-map and lesion-presence pseudo-labels).
# Set `data_path` (and `filelist`) in cfgs/pretrain/clifo_stage1.yaml first. The pretraining data is not released.
# The weights are initialised from the pretrained MIRAGE ViT-B/32 (`weights:` in the config).
set -euo pipefail
cd "$(dirname "$0")/.."
torchrun --nproc_per_node=2 run_pretraining_bscan.py \
    --config cfgs/pretrain/clifo_stage1.yaml \
    --num_workers 32 \
    --clip_grad 5.0
# The final B-scan encoder is the checkpoint of epoch 14 (index 014); copy it to _weights/CLIFO.pth.
