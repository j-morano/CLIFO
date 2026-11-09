import os
import os.path
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, cast, Union
import joblib

import torch
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_nonclass_dataset(
        directory: str,
        args,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    print(f"Making non-class dataset from {directory}")
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    target_dir = directory
    assert os.path.isdir(target_dir), target_dir
    for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                # Only include files that are in the fsids list
                if args.fsids is not None:
                    if Path(path).stem in args.fsids:
                        item = path, 0
                        instances.append(item)
                else:
                    item = path, 0
                    instances.append(item)
    return instances


def normalize_to_0_1(sample: np.ndarray) -> np.ndarray:
    """Normalize to 0-1 range data with any range (positive values)."""
    return (sample - np.min(sample)) / (np.max(sample) - np.min(sample))


class DatabankNavigator:
    def __init__(self, path):
        print(f"Loading databank navigator from {path}...")
        # CRITICAL: mmap_mode='r' makes the huge arrays shareable across processes
        data = joblib.load(path, mmap_mode='r')

        self.patients = data['patients']      # List of (id, start, end)
        self.volumes = data['volumes']        # List of (fsid, start, end)
        self.thicknesses = data['thicknesses']
        self.positions = data['positions']
        self.bscan_ids = data['bscan_ids']
        self.vol_meta = data['vol_meta']

        # O(1) Lookups mapping ID to the index in our tuple lists
        self.pat_id_to_idx = {p[0]: i for i, p in enumerate(self.patients)}
        self.fsid_to_vol_idx = {v[0]: i for i, v in enumerate(self.volumes)}

    def get_fsids_by_patient(self, patient_id):
        if patient_id not in self.pat_id_to_idx:
            return []

        p_idx = self.pat_id_to_idx[patient_id]
        _, v_start, v_end = self.patients[p_idx]

        return [self.volumes[i][0] for i in range(v_start, v_end)]

    def get_full_volume_by_fsid(self, fsid, format_as_str=True):
        if fsid not in self.fsid_to_vol_idx:
            return None

        v_idx = self.fsid_to_vol_idx[fsid]
        _, b_start, b_end = self.volumes[v_idx]

        raw_ids = self.bscan_ids[b_start:b_end]
        bscan_ids = [f"{int(x):03d}" for x in raw_ids] if format_as_str else raw_ids

        return {
            'fsid': fsid,
            'laterality': int(self.vol_meta[v_idx][0]),
            'device': int(self.vol_meta[v_idx][1]),
            'width_mm': self.vol_meta[v_idx][2],
            'thicknesses': self.thicknesses[b_start:b_end],
            'positions': self.positions[b_start:b_end],
            'bscan_ids': bscan_ids,
            'count': b_end - b_start
        }

    def get_bscan_info(self, fsid, bscan_id):
        if fsid not in self.fsid_to_vol_idx:
            return None

        v_idx = self.fsid_to_vol_idx[fsid]
        _, b_start, b_end = self.volumes[v_idx]

        # Slice the ID array and use binary search (O(log N))
        vol_bscan_ids = self.bscan_ids[b_start:b_end]
        target_id = int(bscan_id)
        offset = np.searchsorted(vol_bscan_ids, target_id)

        if offset >= len(vol_bscan_ids) or vol_bscan_ids[offset] != target_id:
            return None

        global_idx = b_start + offset
        meta = self.vol_meta[v_idx]

        return {
            'thicknesses': self.thicknesses[global_idx],
            'position': self.positions[global_idx],
            'bscan_id': f"{target_id:03d}",
            'laterality': int(meta[0]),
            'device': int(meta[1]),
            'width_mm': meta[2],
            'fsid': fsid
        }

    def get_random_bscan_from_patient_idx(self, patient_idx):
        _, v_start, v_end = self.patients[patient_idx]

        v_idx = np.random.randint(v_start, v_end)
        _, b_start, b_end = self.volumes[v_idx]

        b_idx = np.random.randint(b_start, b_end)
        meta = self.vol_meta[v_idx]

        return {
            'thicknesses': self.thicknesses[b_idx],
            'position': self.positions[b_idx],
            'bscan_id': f"{int(self.bscan_ids[b_idx]):03d}",
            'laterality': int(meta[0]),
            'device': int(meta[1]),
            'width_mm': meta[2],
            'fsid': self.volumes[v_idx][0]
        }

    def get_random_volume_from_patient_idx(self, patient_idx):
        _, v_start, v_end = self.patients[patient_idx]
        v_idx = np.random.randint(v_start, v_end)
        return self.get_full_volume_by_fsid(self.volumes[v_idx][0])


class MultiTaskPatientBalancedDatasetFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        tasks: List[str],
        loader: Callable[[str], Any],
        args,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform)
        assert args is not None
        self.tasks = tasks
        self.args = args
        self.loader = loader

        self.navigator = DatabankNavigator(args.filelist)

        print('🚨 Number of patients in filelist:', len(self.navigator.patients))
        print('🚨 Number of B-scans in filelist:', len(self.navigator.bscan_ids))

    def read_item(self, subpath: Union[Path, str]) -> dict:
        sample_dict = {}
        path = Path(self.root, subpath)
        sample_all = np.array(Image.open(path))
        # 0: B-scan, 1: layers, 2: lesions
        for task in self.tasks:
            if task == 'bscan':
                sample = sample_all[:, :, 0].astype(np.float32) / 255.0
            elif task == 'bscanlayermap':
                sample = sample_all[:, :, 1].astype(int)
                # IMPORTANT: Treat the sclera as background
                sample[sample == 12] = 0
                sample_les = sample_all[:, :, 2].astype(int) - 1
                sample[sample_les > 0] = sample_les[sample_les > 0]
            sample_dict[task] = sample
        return sample_dict

    def get_volume_info(self, index: int):
        return self.navigator.get_random_bscan_from_patient_idx(index)

    def __getitem__(self, index: int) -> Tuple:
        volume = self.get_volume_info(index)
        thicknesses = torch.tensor(volume['thicknesses'])
        thicknesses[1::2] = torch.clamp(thicknesses[1::2], max=2.0) * 3

        fsid = volume['fsid']
        bscan_id = volume['bscan_id']
        info = {
            'fsid': fsid,
            # Convert laterality and device to int
            'laterality': volume['laterality'],
            'device': volume['device'],
            'width_mm': volume['width_mm'],
            'position': volume['position'],
            'thicknesses': thicknesses,
        }

        part = fsid.split('-')[2]
        subdir_1 = part[:2]
        subdir_2 = part[2:3]
        subpath = Path(self.root) / subdir_1 / subdir_2 / fsid / f'{bscan_id}.png'
        sample_dict = self.read_item(subpath)

        if self.transform is not None:
            # Add laterality to sample_dict
            sample_dict['laterality'] = info['laterality']
            sample_dict = self.transform(sample_dict)
            # Remove from sample_dict and add to info
            info['laterality'] = sample_dict.pop('laterality')
            if 'lesion_presence' in sample_dict:
                 info['lesion_presence'] = sample_dict.pop('lesion_presence')

        return sample_dict, info

    def __len__(self) -> int:
        return len(self.navigator.patients)
