#!/usr/bin/env bash
# Volume-level retrieval (CLIFO vs OCTCube) and feature-space figures, from the cached 3D features.
set -euo pipefail
source "$(dirname "$0")/common.sh"
root="$CLIFO_OUTPUT/cls_3d/SOTA_comparison_features/0"
for ds in "${DATASETS_3D[@]}"; do
    for model in octcube3d_60_linear_w_cls_patch CLIFO3D_linear_w_2avg; do
        python feature_analysis.py get_results -r "$root/$ds/$model" --metrics_only  # retrieval metrics -> __outputs_feature_analysis/retrieval.csv
        python feature_analysis.py run_tsne_intelli -r "$root/$ds/$model"  # t-SNE
    done
done
# Class dendrograms (OCTAVE)
for model in octcube3d_60_linear_w_cls_patch CLIFO3D_linear_w_2avg; do
    python feature_analysis.py analyze_class_similarity -r "$root/OCTAVE/$model"
done
python gather_results/retrieval.py
