from pathlib import Path
from collections import OrderedDict
import argparse

import pandas as pd

from paths import OUTPUT, RAW_RESULTS


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='2D')
args = parser.parse_args()

selected_seeds = [
    '0', '1', '2', '3', '4', '5', '6',
]
IGNORE_DATASETS = [
    '9C_complete',
]
selected_datasets = [
]

if args.experiment == '2d':
    model_mapping = OrderedDict({
        "retfound_linear_w_patch": "RETFound",
        "visionfm-base_linear_w_cls_patch": "VisionFM",
        "urfound_linear_w_cls_patch": "UrFound",
        "octcube_linear_w_cls_patch": "OCTCube",
        "miragev1-large_linear_w_patch": "MIRAGE",
        "CLIFO_linear_w_patch_lesion": "CLIFO",
    })

elif args.experiment == '3d':
    model_mapping = OrderedDict({
        "octcube3d_60_linear_w_cls_patch": "OCTCube (60 B-scans)",
        "CLIFO3D_60_linear_w_2avg": "CLIFO (60 B-scans)",
        "CLIFO3D_linear_w_2avg": "CLIFO (all B-scans)",
    })

elif args.experiment == 'chd':
    model_mapping = OrderedDict({
        "octcube-chd_linear_w_cls_patch": "OCTCube",
        "miragev1-large-chd_linear_w_patch": "MIRAGE",
        "retfound-chd_linear_w_patch": "RETFound",
        "urfound-chd_linear_w_cls_patch": "UrFound",
        "visionfm-base-chd_linear_w_cls_patch": "VisionFM",
        "CLIFO_ChD_linear_w_patch_lesion": "CLIFO",
    })
    selected_seeds = [str(i) for i in range(14)]


pattern = "*/*/*/*/test_eval.csv"


def map_model_name(folder_name):
    for key, pretty_name in model_mapping.items():
        if key.lower() == folder_name.lower():
            return pretty_name
    return folder_name


def gather_results(root_dir):
    data_list = []
    root_path = Path(root_dir)
    print(f"Looking for files in {root_path} with pattern {pattern}")

    for csv_path in root_path.glob(pattern):
        run_id = csv_path.parts[-2]
        dataset = csv_path.parts[-5]
        model_folder = csv_path.parts[-4]

        if selected_seeds and run_id not in selected_seeds:
            continue
        if selected_datasets:
            if dataset not in selected_datasets:
                continue
        elif dataset in IGNORE_DATASETS:
            continue

        model_display_name = map_model_name(model_folder)
        if model_display_name not in model_mapping.values():
            continue

        try:
            df = pd.read_csv(csv_path)
            best_row = df[df['Epoch'] == 'Best'].copy()
            for col in best_row.columns:
                if col != 'Epoch':
                    best_row[col] = pd.to_numeric(best_row[col], errors='coerce')

            best_row['Run'] = run_id
            best_row['Dataset'] = dataset
            best_row['Model'] = model_display_name

            data_list.append(best_row)
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    if not data_list:
        print("No data found.")
        return None

    full_df = pd.concat(data_list, ignore_index=True)
    full_df = full_df[full_df['Model'].isin(model_mapping.values())]

    # Sanity check: same number of data points per model
    for model_name in model_mapping.values():
        count = (full_df['Model'] == model_name).sum()
        print(f"Model: {model_name.replace(chr(10), ' ')}, Data Points: {count}")

    return full_df


if args.experiment == '3d':
    print("3D MODELS")
    root_dir = OUTPUT / 'cls_3d' / 'SOTA_comparison_features' / '0'
elif args.experiment == '2d':
    print("2D MODELS")
    root_dir = OUTPUT / 'cls' / 'SOTA_comparison_features' / '0'
elif args.experiment == 'chd':
    root_dir = OUTPUT / 'cls' / 'SOTA_change_detection' / '0'

full_df = gather_results(Path(root_dir))

if full_df is not None:
    save_fn = RAW_RESULTS / f"{args.experiment}_raw_results.csv"
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(save_fn, index=False)
    print(f"\nSaved {len(full_df)} rows to {save_fn}")
