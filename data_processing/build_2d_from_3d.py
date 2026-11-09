"""Builds the 2D B-scan classification images of GAMMA, Harvard_Glaucoma, Noor_Eye_Hospital and OLIVES_baseline
from the corresponding 3D volumes (see new_3d_downstream.py), using the file names stored in splits/classification_2d.json.

In these datasets each 2D image is a single B-scan of a volume: the number at the end of the file name is the index of
the B-scan in the volume (0-based, except for Noor_Eye_Hospital, where it is 1-based).

Usage:
    python build_2d_from_3d.py --splits ../splits/classification_2d.json \
        --root3d <path>/Classification_3D --out2d <path>/Classification
"""
import argparse
import json
from pathlib import Path

import numpy as np
from skimage import io

# dataset -> offset between the number in the file name and the B-scan index in the volume
DATASETS = {
    'GAMMA': 0,
    'Harvard_Glaucoma': 0,
    'Noor_Eye_Hospital': 1,
    'OLIVES_baseline': 0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--root3d', type=Path, required=True)
    parser.add_argument('--out2d', type=Path, required=True)
    args = parser.parse_args()
    splits = json.loads(args.splits.read_text())['datasets']
    for dataset, offset in DATASETS.items():
        for split, classes in splits[dataset].items():
            for cls_name, fns in classes.items():
                for fn in fns:
                    stem, idx = Path(fn).stem.rsplit('_', 1)
                    vol = np.load(args.root3d / dataset / split / cls_name / f'{stem}.npz')['vol']
                    out_fn = args.out2d / dataset / split / cls_name / fn
                    out_fn.parent.mkdir(parents=True, exist_ok=True)
                    io.imsave(out_fn, vol[int(idx) - offset])
                    print(out_fn)


if __name__ == '__main__':
    main()
