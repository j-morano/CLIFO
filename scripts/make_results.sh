#!/usr/bin/env bash
# Gathers the results and generates the tables and figures of the paper (into eval/__gen_results/).
set -euo pipefail
source "$(dirname "$0")/common.sh"
for exp in 2d 3d chd; do
    python gather_results/cls.py --experiment "$exp"
done
python gather_results/seg.py --experiment sota
python gather_results/make_plots.py                 # tables (LaTeX), box plots and statistics
python gather_results/build_full_heatmap.py          # per-dataset heatmap
python gather_results/build_summary_heatmap.py       # per-disease summary heatmap
# Qualitative segmentation figure for chosen test B-scans: python qualitative_seg_results.py <dataset>/<file> ...
python explainability.py                                # [Lesions]-token attention maps (needs _weights/CLIFO.pth)
