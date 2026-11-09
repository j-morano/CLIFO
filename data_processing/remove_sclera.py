"""Segmentation datasets without the sclera label.

remove_sclera_labels: AROI, Duke_DME and Duke_iAMD_labeled -> <dataset>_nosclera. The sclera label is mapped to the
    background value and removed from INFO.json.
filter_duke_iamd: Duke_iAMD_labeled_nosclera -> Duke_iAMD_labeled_nosclera_<threshold>. Keeps only the B-scans in
    which more than <threshold> % of the pixels are labeled.

Usage:
    python remove_sclera.py remove_sclera_labels
    python remove_sclera.py filter_duke_iamd --threshold 70
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from skimage import io

DATASETS_ROOT = Path(os.environ.get('DATASETS_ROOT', './_datasets'))  # processed datasets
SEG_ROOT = DATASETS_ROOT / 'Segmentation'

# Label of the sclera in each dataset, and the value it is replaced by
SCLERA_LABELS = {
    'AROI': 'Under BM',
    'Duke_DME': 'BM',
    'Duke_iAMD_labeled': 'Below BM',
}
REPLACEMENT_VALUES = {
    'AROI': 0,
    'Duke_DME': 0,
    'Duke_iAMD_labeled': 51,
}


def remove_sclera_labels():
    for dataset, sclera_label in SCLERA_LABELS.items():
        dataset_path = SEG_ROOT / dataset
        new_dataset_path = SEG_ROOT / f'{dataset}_nosclera'
        with open(dataset_path / 'INFO.json', 'r') as f:
            info = json.load(f)
        # INFO.json: {"0": {"value": 0, "label": "Background"}, ...}
        new_info = {k: v for k, v in info.items() if v['label'] != sclera_label}
        value_to_remove = next(v['value'] for v in info.values() if v['label'] == sclera_label)

        # Duke_iAMD_labeled has no train/val/test folders
        splits = [''] if dataset == 'Duke_iAMD_labeled' else ['train', 'val', 'test']
        for split in splits:
            split_path = dataset_path / split
            assert split_path.exists(), split_path
            tgt_bscan_path = new_dataset_path / split / 'bscan'
            if tgt_bscan_path.exists():
                shutil.rmtree(tgt_bscan_path)
            shutil.copytree(split_path / 'bscan', tgt_bscan_path)
            for semseg_fn in sorted((split_path / 'semseg').iterdir()):
                semseg = io.imread(semseg_fn)
                new_semseg = semseg.copy()
                new_semseg[semseg == value_to_remove] = REPLACEMENT_VALUES[dataset]
                tgt_fn = new_dataset_path / split / 'semseg' / semseg_fn.name
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                io.imsave(tgt_fn, new_semseg)
        with open(new_dataset_path / 'INFO.json', 'w') as f:
            json.dump(new_info, f, indent=4)
        print(f'{dataset} -> {new_dataset_path}')


def filter_duke_iamd(threshold=70):
    src_dir = SEG_ROOT / 'Duke_iAMD_labeled_nosclera'
    # 50 % was the first version of the filtered dataset, named "cleaned"
    suffix = 'cleaned' if threshold == 50 else f'{threshold}'
    tgt_dir = SEG_ROOT / f'Duke_iAMD_labeled_nosclera_{suffix}'
    tgt_dir.mkdir(exist_ok=True)
    shutil.copy(src_dir / 'INFO.json', tgt_dir / 'INFO.json')
    copied = 0
    total = 0
    for semseg_fn in sorted((src_dir / 'semseg').iterdir()):
        semseg = io.imread(semseg_fn)
        percentage_labeled = (semseg != 0).sum() / semseg.size * 100
        if percentage_labeled > threshold:
            for folder in ['semseg', 'bscan']:
                tgt_fn = tgt_dir / folder / semseg_fn.name
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_dir / folder / semseg_fn.name, tgt_fn)
            copied += 1
        total += 1
    print(f'Copied {copied}/{total} B-scans with more than {threshold}% labeled pixels to {tgt_dir}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function', choices=['remove_sclera_labels', 'filter_duke_iamd'])
    parser.add_argument('--threshold', type=float, default=70, help='Minimum percentage of labeled pixels (filter_duke_iamd).')
    args = parser.parse_args()
    if args.function == 'remove_sclera_labels':
        remove_sclera_labels()
    else:
        filter_duke_iamd(args.threshold)
