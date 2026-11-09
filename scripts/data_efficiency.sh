#!/usr/bin/env bash
# Few-shot (labelled samples per class) linear probing on the 3D features, and the efficiency benchmark (Fig. 3c, 3d).
set -euo pipefail
source "$(dirname "$0")/common.sh"
root="$CLIFO_OUTPUT/cls_3d/SOTA_comparison_features/0"
for ds in "${DATASETS_3D[@]}"; do
    for model in octcube3d_60_linear_w_cls_patch CLIFO3D_linear_w_2avg; do
        for n in 1 2 5 7 10 15 20 30 10000; do
            for s in "${SEEDS[@]}"; do
                python -m feature_probing -r "$root/$ds/$model" --n_per_class "$n" --seed "$s" --val_metric mcc --balance
            done
        done
    done
done
python plot_data_efficiency.py --root "$root"          # Fig. 3c and table
# Inference time and memory against the number of B-scans (Fig. 3d)
for model in CLIFO3D octcube3d_60; do
    python benchmark_efficiency.py --weights "$model"
done
python plot_efficiency.py                              # Fig. 3d and table
