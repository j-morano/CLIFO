#!/usr/bin/env bash
# Linear probing on the cached features. Usage: scripts/probe.sh {2d|3d|chd}
set -euo pipefail
source "$(dirname "$0")/common.sh"
mode="${1:?usage: $0 2d|3d|chd}"
seeds=("${SEEDS[@]}")
case "$mode" in
    2d)  root="$CLIFO_OUTPUT/cls/SOTA_comparison_features/0" ;;
    chd) root="$CLIFO_OUTPUT/cls/SOTA_change_detection/0"; seeds=("${SEEDS_CHD[@]}") ;;
    3d)  root="$CLIFO_OUTPUT/cls_3d/SOTA_comparison_features/0" ;;
    *) echo "unknown mode: $mode"; exit 1 ;;
esac
for dir in "$root"/*/*/; do
    for s in "${seeds[@]}"; do
        python -m feature_probing -r "$dir" --seed "$s" --val_metric mcc --balance
    done
done
