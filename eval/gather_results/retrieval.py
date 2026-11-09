"""Gathers the retrieval results written by `feature_analysis.py get_results` (./__outputs_feature_analysis/retrieval.csv)
into the table used by the plotting scripts (__raw_results/retrieval_raw_results.csv).

Usage (from eval/): PYTHONPATH=. python gather_results/retrieval.py
"""
import pandas as pd

from paths import RAW_RESULTS

NAMES = {
    'CLIFO3D_linear_w_2avg': 'CLIFO',
    'octcube3d_60_linear_w_cls_patch': 'OCTCube',
}

df = pd.read_csv('./__outputs_feature_analysis/retrieval.csv')
df['model_name'] = df['model_name'].map(NAMES).fillna(df['model_name'])
# `get_results` appends: keep the last result of each (model, dataset)
df = df.drop_duplicates(subset=['model_name', 'dataset_name'], keep='last')
RAW_RESULTS.mkdir(parents=True, exist_ok=True)
save_fn = RAW_RESULTS / 'retrieval_raw_results.csv'
df.to_csv(save_fn, index=False)
print(f'Saved {len(df)} rows to {save_fn}')
