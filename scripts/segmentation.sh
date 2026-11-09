#!/usr/bin/env bash
# Segmentation: trains the heads on the frozen features, evaluates, and gathers the results.
set -euo pipefail
source "$(dirname "$0")/common.sh"
out="$CLIFO_OUTPUT/seg"
for s in "${SEEDS[@]}"; do
    for m in "${MODELS_SEG[@]}"; do
        for ds in "${DATASETS_SEG[@]}"; do
            python run_seg_tuning.py --version SOTA_comparison_new --adapter convnext --epochs 250 --seed "$s" \
                --num_workers 8 --pool seg --weights "$m" --data_set "./_datasets/Segmentation/$ds" --base_output_dir "$out"
        done
        # AROI -> Duke iAMD (external test set)
        python run_seg_tuning.py --version SOTA_comparison_new --adapter convnext --epochs 250 --seed "$s" \
            --num_workers 8 --pool seg --weights "$m" --data_set ./_datasets/Segmentation/AROI_nosclera \
            --test_only --infer_only --test_data ./_datasets/Segmentation/Duke_iAMD_labeled_nosclera_70 --base_output_dir "$out"
    done
    for dir in "$out/SOTA_comparison_new/$s"/*/*/; do
        python run_seg_eval.py --model_path "$dir"
    done
    for dir in "$out/SOTA_comparison_new/$s"/AROI_nosclera/*/; do
        python run_seg_eval.py --external Duke_iAMD_labeled_nosclera_70 --model_path "$dir"
    done
done
python gather_results/seg.py --experiment sota
