#!/usr/bin/env bash
# Caches the (frozen) features of every model and dataset. Usage: scripts/extract_features.sh {2d|3d|chd}
# 3d runs the volume models (also used for retrieval); chd is change detection (MARIO).
set -euo pipefail
source "$(dirname "$0")/common.sh"
mode="${1:?usage: $0 2d|3d|chd}"
extra=()
case "$mode" in
    2d)  data_root=_datasets/Classification/;    version=SOTA_comparison_features; out=cls;    datasets=("${DATASETS_2D[@]}"); models=("${MODELS_2D[@]}") ;;
    chd) data_root=_datasets/Classification/;    version=SOTA_change_detection;    out=cls;    datasets=(MARIO_T1);            models=("${MODELS_CHD[@]}") ;;
    3d)  data_root=_datasets/Classification_3D/; version=SOTA_comparison_features; out=cls;    datasets=("${DATASETS_3D[@]}"); models=("${MODELS_3D[@]}"); extra=(--three_d) ;;
    *) echo "unknown mode: $mode"; exit 1 ;;
esac
for ds in "${datasets[@]}"; do
    for m in "${models[@]}"; do
        read -r weights pool <<< "$m"
        python -m run_cls_tuning --version "$version" --seed 0 --weights "$weights" --pool "$pool" \
            --data_root "$data_root" --data_set "$ds" --smoothing 0.0 --balance --device cuda \
            --cache_features --base_output_dir "$CLIFO_OUTPUT/$out" "${extra[@]}"
    done
done
