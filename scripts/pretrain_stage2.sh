#!/usr/bin/env bash
# Stage 2: volume feature encoder (VFE), trained with a vision-language objective on the clinical reports.
# Requires the stage-1 weights in _weights/CLIFO.pth and the report data (see docs/stage2_reports.md).
#   VLM_DATA_ROOT  folder with the volumes and the report file
set -euo pipefail
cd "$(dirname "$0")/.."
: "${VLM_DATA_ROOT:?set VLM_DATA_ROOT}"
python -m run_pretraining_vfe main_train \
    --data_root "$VLM_DATA_ROOT" \
    --device cuda \
    --num_epochs 10 \
    --version clifo \
    --num_queries 16 \
    --resampler xattn_v2 \
    --lesion_pool xattn_v2
# The VFE used for downstream tasks is the checkpoint of epoch 10; copy it to _weights/CLIFO_VFE.pt, e.g.
#   __posttraining/clifo/r-xattn_v2_lp-xattn_v2_nq-16_as-32_sk-True_fw-True/010.pt
# Report-generation evaluation (BLEU, ROUGE-L; needs the `report` extra):
#   python -m run_pretraining_vfe main_val --data_root "$VLM_DATA_ROOT" --device cuda --weights _weights/CLIFO_VFE.pt
