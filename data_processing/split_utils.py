"""Helpers to read the static splits in ../splits/*.json.

The builders look up every file they produce in the split files, so the splits do not depend on random seeds or on
directory listings. Set CLIFO_SPLITS to use another splits directory.
"""
import json
import os
from pathlib import Path

SPLITS_DIR = Path(os.environ.get('CLIFO_SPLITS', Path(__file__).resolve().parent.parent / 'splits'))


def _load(task):
    with open(SPLITS_DIR / f'{task}.json') as f:
        return json.load(f)['datasets']


class ClsSplit:
    """Classification splits: file name -> (split, class).

    task: 'classification_2d', 'classification_3d' or 'change_detection'.
    """

    def __init__(self, task, dataset):
        self.dataset = dataset
        self.items = {}
        for split, classes in _load(task)[dataset].items():
            for cls_name, fns in classes.items():
                for fn in fns:
                    assert fn not in self.items, f'{dataset}: duplicated file name {fn}'
                    self.items[fn] = (split, cls_name)
        self.seen = set()

    def get(self, fn):
        """(split, class) of a file name, or None if it is not in the splits."""
        fn = str(fn)
        if fn in self.items:
            self.seen.add(fn)
            return self.items[fn]
        return None

    def __iter__(self):
        """Yields (split, class, file name)."""
        for fn, (split, cls_name) in self.items.items():
            self.seen.add(fn)
            yield split, cls_name, fn

    def report(self):
        missing = sorted(set(self.items) - self.seen)
        print(f'{self.dataset}: {len(self.seen)}/{len(self.items)} files of the split were produced')
        if missing:
            print(f'  WARNING: {len(missing)} files not produced, e.g. {missing[:3]}')


class SegSplit:
    """Segmentation splits: B-scan file name -> list of splits (the *_labeled test sets are subsets of test)."""

    def __init__(self, dataset):
        self.dataset = dataset
        entry = _load('segmentation')[dataset]
        self.labels = entry['labels']
        self.folders = {}  # split -> list of folders with masks (semseg, semseg_2, ...)
        self.items = {}
        for split, folders in entry['splits'].items():
            self.folders[split] = [k for k in folders if k != 'bscan']
            for fn in folders['bscan']:
                self.items.setdefault(fn, []).append(split)
        self.seen = set()

    def get(self, fn):
        """List of the splits that contain a B-scan file name (empty if none)."""
        fn = str(fn)
        if fn in self.items:
            self.seen.add(fn)
        return self.items.get(fn, [])

    def write_info(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'INFO.json', 'w') as f:
            json.dump(self.labels, f, indent=4)

    def report(self):
        missing = sorted(set(self.items) - self.seen)
        print(f'{self.dataset}: {len(self.seen)}/{len(self.items)} B-scans of the split were produced')
        if missing:
            print(f'  WARNING: {len(missing)} B-scans not produced, e.g. {missing[:3]}')
