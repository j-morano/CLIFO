"""Locations of the data and of the results. Override them with environment variables.

CLIFO_DATASETS    root of the processed datasets (Classification/, Classification_3D/, Segmentation/)
CLIFO_OUTPUT      root of the outputs of feature extraction, probing and segmentation (cls/, cls_3d/, seg/)
CLIFO_RAW_RESULTS gathered result tables (CSV files read by the plotting scripts)
CLIFO_GEN_RESULTS generated tables and figures
"""
import os
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATASETS = Path(os.environ.get('CLIFO_DATASETS', EVAL_DIR / '_datasets'))
OUTPUT = Path(os.environ.get('CLIFO_OUTPUT', EVAL_DIR / '__output'))
RAW_RESULTS = Path(os.environ.get('CLIFO_RAW_RESULTS', EVAL_DIR / '__raw_results'))
GEN_RESULTS = Path(os.environ.get('CLIFO_GEN_RESULTS', EVAL_DIR / '__gen_results'))
