import os
from pathlib import Path
import argparse
from glob import glob
import re
from collections import OrderedDict

from skimage import io
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import label

from split_utils import ClsSplit, SegSplit

RAW_DATA_ROOT = Path(os.environ.get('RAW_DATA_ROOT', '/path/to/raw_datasets'))  # downloads of the original datasets
DATASETS_ROOT = Path(os.environ.get('DATASETS_ROOT', './_datasets'))  # processed datasets (output)


BASE_DATA_PATH = RAW_DATA_ROOT


TGT_BASE_PATH = DATASETS_ROOT / 'Classification_3D'
TGT_SEG_DIR = DATASETS_ROOT / 'Segmentation'


def convert_to_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to grayscale using luminosity method
        return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype('uint8')
    elif len(image.shape) == 2:
        # Image is already grayscale
        return image
    else:
        raise ValueError("Unsupported image format")


def oimhs():
    path_dir = BASE_DATA_PATH / 'OIMHS_dataset/Images'
    tgt_path = TGT_BASE_PATH / 'OIMHS'
    S = ClsSplit('classification_3d', 'OIMHS')
    for split, cls_name, fn in S:
        patient_id, eye_id = Path(fn).stem.split('_')
        eye_dir = path_dir / eye_id
        imgs = list(eye_dir.glob('*.png'))
        assert imgs and len(imgs) > 0
        scan_nums = sorted([int(Path(img_fn).stem) for img_fn in imgs])
        volume = []
        volume_seg = []
        for scan_num in scan_nums:
            img = io.imread(eye_dir / f'{scan_num}.png')
            # seg is right half, MH is red
            bscan = img[:, :img.shape[1] // 2]
            seg = img[:, img.shape[1] // 2:]
            bscan = convert_to_grayscale(bscan)
            volume.append(bscan)
            volume_seg.append(seg)
        tgt_fn = tgt_path / split / cls_name / fn
        print(' ', tgt_fn)
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tgt_fn, vol=np.array(volume), seg=np.array(volume_seg))
    S.report()


def poagd():
    path_dir = BASE_DATA_PATH / 'POAG_detection/glaucoma_detection'
    tgt_base = TGT_BASE_PATH / 'POAGD'
    S = ClsSplit('classification_3d', 'POAGD')
    for split, cls_name, fn in S:
        volume = np.load(path_dir / f'{Path(fn).stem}.npy')
        tgt_fn = tgt_base / split / cls_name / fn
        print(' ', tgt_fn)
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tgt_fn, vol=volume)
    S.report()


def umn():
    base_path = BASE_DATA_PATH / 'FoundOPTIMA_Downstream' / 'UMN'
    data_fn = 'UMN_Dataset.mat'
    target_path = TGT_BASE_PATH / 'UMN'
    target_path.mkdir(exist_ok=True)
    # IMPORTANT: the B-scans are not ordered.

    for cls_name in base_path.iterdir():
        if not cls_name.is_dir():
            continue
        dataset = loadmat(cls_name / data_fn)
        all_subjects = dataset['AllSubjects']
        manual_fluid_1 = dataset['ManualFluid1']
        manual_fluid_2 = dataset['ManualFluid2']
        for i in range(len(all_subjects[0])):
            patient = str(i).zfill(3)
            class_name = cls_name.stem.lower()
            volume = all_subjects[0][i].transpose(2, 0, 1)
            volume_mf1 = manual_fluid_1[0][i].transpose(2, 0, 1)
            volume_mf2 = manual_fluid_2[0][i].transpose(2, 0, 1)
            assert volume.shape == volume_mf1.shape == volume_mf2.shape
            idd = f'{class_name}_{patient}'
            tgt_fn = target_path / class_name / f'{idd}.npz'
            print(' ', tgt_fn)
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tgt_fn, vol=volume, seg1=volume_mf1, seg2=volume_mf2)


def harvardglaucoma():
    base_path = BASE_DATA_PATH / 'FoundOPTIMA_Downstream/Harvard_Glaucoma_Detection/Harvard Glaucoma Detection and Progression with 1000 Samples (Harvard-GDP1000)/Bscan'
    tgt_base = TGT_BASE_PATH / 'Harvard_Glaucoma'
    S = ClsSplit('classification_3d', 'Harvard_Glaucoma')
    for split, cls_name, fn in S:
        volume = np.load(base_path / fn)['bscans']
        save_path = tgt_base / split / cls_name / fn
        print(' ', save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, vol=volume)
    S.report()


def remove_white_part(img):
    """
    Args:
        img: HxW numpy array, range 0-255
    """
    # Pad image with 1s
    padded = np.pad(img, ((16, 16), (16, 16)), mode='constant', constant_values=255)
    # Get biggest connected component of pixels with value > 250
    mask = padded > 250
    labeled, num_features = label(mask)
    largest_component = 0
    largest_size = 0
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if size > largest_size:
            largest_size = size
            largest_component = i
    # Put to 0
    padded[labeled == largest_component] = 0
    # Remove padding
    return padded[16:-16, 16:-16]


def dukes():
    base_path = BASE_DATA_PATH / 'FoundOPTIMA_Downstream' / 'Duke_Srinivasan' / 'Publication_Dataset'
    target_path = TGT_BASE_PATH / 'DukeSrinivasan'
    class_names_map = {
        'NORMAL': 'Normal',
        'AMD': 'AMD',
        'DME': 'DME',
    }
    for volume_fn in base_path.iterdir():
        for cls_ in class_names_map:
            if cls_ in volume_fn.stem:
                cls_name = class_names_map[cls_]
                break
        c_base_path = (volume_fn / 'TIFFs' / '8bitTIFFs')
        img_fns = c_base_path.glob('*.tif*')
        img_nums = sorted([int(fn.stem) for fn in img_fns])
        volume = []
        for img_num in img_nums:
            img_fn = c_base_path / f'0{img_num}.tif'
            img = io.imread(img_fn)
            img = convert_to_grayscale(img)
            img = remove_white_part(img)
            volume.append(img)
        volume = np.array(volume)
        tgt_fn = target_path / cls_name / f'{volume_fn.stem}.npz'
        print(' ', tgt_fn)
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tgt_fn, vol=volume)


def noor():
    base_path = BASE_DATA_PATH / 'FoundOPTIMA_Downstream' / 'Noor_Eye_Hospital' / 'Data'
    target_path = TGT_BASE_PATH / 'Noor_Eye_Hospital'
    S = ClsSplit('classification_3d', 'Noor_Eye_Hospital')
    for cls_name in base_path.iterdir():
        for case in cls_name.iterdir():
            table = str.maketrans({'(': None, ')': None, ' ': '_'})
            fn = f'{case.stem.translate(table)}.npz'
            r = S.get(fn)
            if r is None:
                continue
            img_fns = list(case.glob('*.TIFF')) + list(case.glob('*.PNG'))
            assert img_fns and len(img_fns) > 0
            nums = []
            for img_fn in img_fns:
                num = int(re.search(r"(\d+)", img_fn.stem).group(1))
                img = convert_to_grayscale(io.imread(img_fn))
                nums.append((num, img))
            nums = sorted(nums, key=lambda x: x[0])
            volume = np.array([img for num, img in nums])
            tgt_fn = target_path / r[0] / r[1] / fn
            print(' ', tgt_fn)
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tgt_fn, vol=volume)
    S.report()


def octmanualdelineations():
    import eyepy

    # https://iacl.ece.jhu.edu/index.php?title=Resources
    path = RAW_DATA_ROOT / 'OCT_Manual_Delineations-2018_June_29_b/OCT_Manual_Delineations-2018_June_29'
    S = ClsSplit('classification_3d', 'OCT_MD_MS')
    # ID,Age,Gender,Eye,Diagnosis
    for img_fn in sorted((path / 'vol').iterdir()):
        id_ = img_fn.stem.split('_')[0]
        r = S.get(f'{id_}.npz')
        if r is None:
            continue
        vol = eyepy.import_heyex_vol(img_fn).data
        seg_fn = path / 'delineation' / f'{img_fn.stem}.mat'
        seg = None
        try:
            seg1 = loadmat(seg_fn)['bd_pts']
            seg = seg1
        except KeyError:
            pass
        try:
            seg2 = loadmat(seg_fn)['control_pts']
            seg = seg2
        except KeyError:
            pass
        tgt_fn = TGT_BASE_PATH / 'OCT_MD_MS' / r[0] / r[1] / f'{id_}.npz'
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        print(' ', tgt_fn)
        np.savez_compressed(tgt_fn, vol=vol, seg=seg)
    S.report()


def banglaoct2025():
    base_path_dir = RAW_DATA_ROOT / 'BanglaOCT2025/BanglaOCT2025_Raw_130_Images_Only'
    tgt_path = TGT_BASE_PATH / 'BanglaOCT2025'
    S = ClsSplit('classification_3d', 'BanglaOCT2025')
    for split, cls_name, fn in S:
        patient = Path(fn).stem
        c_path = base_path_dir / cls_name / patient
        assert c_path.exists()
        # All those that match this pattern: oct_c_[ddd].bmp,
        #   d being a digit
        bscan_fns = sorted(c_path.glob('oct_c_[0-9][0-9][0-9].bmp'))
        volume = np.array([convert_to_grayscale(io.imread(bscan_fn)) for bscan_fn in bscan_fns])
        save_fn = tgt_path / split / cls_name / fn
        print(' ', save_fn)
        save_fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_fn, vol=volume)
    S.report()


def get_ordered_bscans(dir: Path):
    extensions = ['.bmp', '.png', '.tif', '.tiff', '.jpeg', '.jpg']
    # Add their uppercase versions
    extensions += [ext.upper() for ext in extensions]
    bscans = []
    for img_fn in dir.iterdir():
        if img_fn.suffix in extensions:
            match = re.search(r'(\d+)(?!.*\d)', img_fn.stem)
            if match:
                num = int(match.group(1))
                bscans.append((num, img_fn))
            else:
                raise ValueError
    return sorted(bscans, key=lambda x: x[0])


def get_ordered_bscans_list(bscan_fns: list):
    bscans = []
    for fn in bscan_fns:
        stem = Path(fn).stem
        match = re.search(r'(\d+)(?!.*\d)', stem)
        if match:
            num = int(match.group(1))
            bscans.append((num, fn))
        else:
            raise ValueError
    return sorted(bscans, key=lambda x: x[0])


def octa500():
    path_dir = RAW_DATA_ROOT / 'OCTA-500'
    # NOTE: Very few cases in total. Merging diseased
    # NORMAL has 160 volumes
    # CNV has 5 volumes
    # DR has 29 volumes
    # AMD has 6 volumes
    S = ClsSplit('classification_3d', 'OCTA500')
    for split, cls_name, fn in S:
        vol_id = Path(fn).stem
        vol_dir = path_dir / 'dataset_3m' / 'OCT' / vol_id
        assert vol_dir.exists()
        bscans = get_ordered_bscans(vol_dir)
        gt_artery = io.imread(path_dir / 'dataset_3m' / 'Label' / 'GT_Artery' / f'{vol_id}.bmp')
        gt_vein = io.imread(path_dir / 'dataset_3m' / 'Label' / 'GT_Vein' / f'{vol_id}.bmp')
        volume = np.array([convert_to_grayscale(io.imread(bscan_fn)) for num, bscan_fn in bscans])
        tgt_fn = TGT_BASE_PATH / 'OCTA500' / split / cls_name / fn
        print(' ', tgt_fn)
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tgt_fn, vol=volume, gt_artery=gt_artery, gt_vein=gt_vein)
    S.report()


def m3ret():
    path_dir = RAW_DATA_ROOT / 'M3Ret'
    S = ClsSplit('classification_3d', 'M3Ret')
    for split, cls_name, fn in S:
        pat_id, laterality = Path(fn).stem.split('_')
        img_pattern = path_dir / 'OCTImages-Macular-v3.4' / f'{str(pat_id).zfill(10)}*' / laterality / 'study1' / 'volume_1*'
        img_fns = glob(str(img_pattern))
        assert img_fns and len(img_fns) > 0
        bscan_fns = get_ordered_bscans_list(img_fns)
        volume = convert_image_list_to_volume(bscan_fns)
        tgt_fn = TGT_BASE_PATH / 'M3Ret' / split / cls_name / fn
        print(' ', tgt_fn)
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tgt_fn, vol=volume)
    S.report()


def convert_image_list_to_volume(bscan_fns: list):
    volume = []
    for num, bscan_fn in bscan_fns:
        bscan = io.imread(bscan_fn)
        bscan_gray = convert_to_grayscale(bscan)
        volume.append(bscan_gray)
    return np.array(volume)


def olives_baseline():
    path_dir = RAW_DATA_ROOT / 'FoundOPTIMA_Downstream/OLIVES/OLIVES'
    tgt_base = TGT_BASE_PATH / 'OLIVES_baseline'
    S = ClsSplit('classification_3d', 'OLIVES_baseline')
    for split, cls_name, fn in S:
        stem = Path(fn).stem
        parts = stem.split('_')
        if 'Prime' in stem:
            study = parts[2] + '_' + parts[3]
            patient = parts[4]
            visit = parts[5]
            laterality = parts[6]
            vol_dir = path_dir / study / patient / visit / laterality
            pattern = r'^[0-9]+\.(png|tif)$'
        else:
            # 0249_0019_TREX_DME_GILA_0249GOS_V1_OS_TREXA.npz
            study = parts[2] + ' ' + parts[3]
            whatever = parts[4]
            patient = parts[5]
            visit = parts[6]
            laterality = parts[7]
            vol_dir = path_dir / study / whatever / patient / visit / laterality
            pattern = r'^TREX.*_[0-9]{6}\.(tif|jpeg)$'
        bscan_fns = [f for f in vol_dir.glob('*.*') if re.match(pattern, f.name)]
        bscan_fns = get_ordered_bscans_list(bscan_fns)
        volume = convert_image_list_to_volume(bscan_fns)
        tgt_fn = tgt_base / split / cls_name / fn
        print(' ', tgt_fn)
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tgt_fn, vol=volume)
    S.report()


def gamma():
    path_dir = RAW_DATA_ROOT / 'GAMMA/Glaucoma_grading/training/multi-modality_images'
    tgt_base = TGT_BASE_PATH / 'GAMMA'
    S = ClsSplit('classification_3d', 'GAMMA')
    for split, cls_name, fn in S:
        vol_id = Path(fn).stem
        bscan_fns = get_ordered_bscans(path_dir / vol_id / vol_id)
        volume = convert_image_list_to_volume(bscan_fns)
        tgt_fn = tgt_base / split / cls_name / fn
        print(' ', tgt_fn)
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tgt_fn, vol=volume)
    S.report()


def octave():
    path_dir = RAW_DATA_ROOT / 'OCTAVE/nnUNet_raw/Dataset001_OCTAVE/imagesTr'
    tgt_base = TGT_BASE_PATH / 'OCTAVE'
    S = ClsSplit('classification_3d', 'OCTAVE')
    for split, cls_name, fn in S:
        disease, iid = Path(fn).stem.split('_')
        vol = io.imread(path_dir / f'OCTAVE_{disease}_{iid}_0000.tif')
        label = io.imread(path_dir.parent / 'labelsTr' / f'OCTAVE_{disease}_{iid}.tif')
        assert vol.shape == label.shape
        new_vol = np.zeros_like(vol)
        for i in range(vol.shape[0]):
            new_vol[i] = remove_white_part(vol[i])
        tgt_fn = tgt_base / split / cls_name / fn
        print(' ', tgt_fn)
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tgt_fn, vol=new_vol, seg=label)
    S.report()


def save_seg_sample(S, split, name, bscan, masks):
    """Saves a B-scan and its masks in <TGT_SEG_DIR>/<dataset>/<split>/{bscan,semseg,...}/<name>.

    S: SegSplit. masks: folder name (semseg, semseg_2, ...) -> mask. The folders of the split (see
    splits/segmentation.json) are the ones that are written.
    """
    folders = S.folders[split]
    assert set(folders) <= set(masks), f'{S.dataset}/{split}: missing masks {set(folders) - set(masks)}'
    tgt_dir = TGT_SEG_DIR / S.dataset / split
    for folder, img in [('bscan', bscan)] + [(f, masks[f]) for f in folders]:
        fn = tgt_dir / folder / name
        fn.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(fn, img)


def rvo_lesion_seg():
    path_dir = RAW_DATA_ROOT / 'RVO-Lesion/Image_Seg'
    S = SegSplit('RVO_Lesion')
    for fn in sorted((path_dir / 'images').glob('*.jpg')):
        vol_id, bscan_id = fn.stem.split('_')
        name = f'{int(vol_id):04d}_{int(bscan_id):03d}.png'
        splits = S.get(name)
        if not splits:
            continue
        img = convert_to_grayscale(io.imread(fn))
        seg = io.imread(path_dir / 'masks' / f'{fn.stem}.png')
        # 1: SRF, 2: IRF
        new_seg = np.zeros_like(seg)
        new_seg[seg == 1] = 127  # SRF
        new_seg[seg == 2] = 255  # IRF
        for split in splits:
            save_seg_sample(S, split, name, img, {'semseg': new_seg})
    S.write_info(TGT_SEG_DIR / S.dataset)
    S.report()


def octave_seg():
    tgt_dir = TGT_SEG_DIR / 'OCTAVE'
    M = 19
    # Labels of the OCTAVE nnU-Net dataset (Dataset001_OCTAVE)
    # Retinal Pigment Epithelium (RPE)
    # Retina (RET)
    # Choroid and Sclera (CHO)
    # Vitreous (VIT)
    # Subretinal Material (SRM)
    # Posterior Hyaloid Membrane (HYA)
    # Retrohyaloid Space (RHS)
    # Epiretinal Membrane (ERM)
    # Sub-ERM Space (SES)
    # Hypertransmission Defect (HTD)
    # Artifact (ART)
    # Retinal/Subretinal Fluid (FLU)
    # Hyperreflective Material (HRM)
    labels = OrderedDict({
        "Background": (0, 0),
        "CHO": (7, 0),
        "VIT": (8, 0),
        "ART": (11, 0),
        "SRM": (1, 1 * M),
        "HRM": (2, 2 * M),
        "FLU": (3, 3 * M),
        "HTD": (4, 4 * M),
        "RPE": (5, 5 * M),
        "RET": (6, 6 * M),
        "HYA": (9, 9 * M),
        "RHS": (10, 10 * M),
        "ERM": (12, 12 * M),
        "SES": (13, 13 * M),
    })
    S = SegSplit('OCTAVE')
    # The volumes are the ones of the 3D classification dataset (see octave())
    for split_3d, cls_name, vol_name in ClsSplit('classification_3d', 'OCTAVE'):
        vol_fn = TGT_BASE_PATH / 'OCTAVE' / split_3d / cls_name / vol_name
        data = np.load(vol_fn)
        vol = data['vol']
        seg = data['seg']
        for label_name, (label_value, new_value) in labels.items():
            seg[seg == label_value] = new_value
        seg = seg.astype(np.uint8)
        for i in range(vol.shape[0]):
            name = f'{vol_fn.stem}_{i:03d}.png'
            for split in S.get(name):
                save_seg_sample(S, split, name, vol[i], {'semseg': seg[i]})
    S.write_info(tgt_dir)
    S.report()


def umn_seg():
    # Volumes from umn(). The 'test' split has all the B-scans of the test volumes, and 'test_labeled' only those with
    #   fluid in both annotations.
    path_dir = TGT_BASE_PATH / 'UMN'
    S = SegSplit('UMN')
    for cls_name in sorted(path_dir.iterdir()):
        for vol_fn in sorted(cls_name.iterdir()):
            data = np.load(vol_fn)
            vol = data['vol']
            seg1 = data['seg1']
            seg1[seg1 == 1] = 255
            seg2 = data['seg2']
            seg2[seg2 == 1] = 255
            for i in range(vol.shape[0]):
                assert len(np.unique(seg1[i])) <= 2 and len(np.unique(seg2[i])) <= 2
                name = f'{vol_fn.stem}_{i:03d}.png'
                for split in S.get(name):
                    save_seg_sample(S, split, name, vol[i], {'semseg': seg1[i], 'semseg_2': seg2[i]})
    S.write_info(TGT_SEG_DIR / S.dataset)
    S.report()


def convert_retouch_seg(seg):
    mapping = {
        1: 85,
        2: 170,
        3: 255,
    }
    for k, v in mapping.items():
        seg[seg == k] = v
    seg = seg.astype(np.uint8)
    return seg


def retouch_test_seg():
    # Only the test sets ('test': all B-scans, 'test_labeled': B-scans with some label in both annotations).
    # The train/val sets are not included here.
    path_dir = RAW_DATA_ROOT / 'RETOUCH'
    S = SegSplit('RETOUCH')
    for dir in sorted((path_dir / 'TestSet-MUW').iterdir()):
        # oct.mhd
        # oct.raw
        # reference.mhd
        # reference.raw
        vol = io.imread(dir / 'oct.mhd')
        seg = convert_retouch_seg(io.imread(dir / 'reference.mhd'))
        seg_2_fn = path_dir / 'TestSet-Radboud' / dir.name / 'reference.mhd'
        seg_2 = convert_retouch_seg(io.imread(seg_2_fn))
        for i in range(vol.shape[0]):
            name = f'{dir.stem}_{i:03d}.png'
            for split in S.get(name):
                save_seg_sample(S, split, name, vol[i], {'semseg': seg[i], 'semseg_Radboud': seg_2[i]})
    S.write_info(TGT_SEG_DIR / S.dataset)
    S.report()


def amd_dme_cls_seg():
    """AMD_DME_3D: 3D classification, 2D classification (central B-scan) and segmentation. 'test' has all the B-scans
    of the test volumes, and 'test_labeled' only those with labels."""
    path_dir = RAW_DATA_ROOT / 'AMD_DME_3D_Dataset'
    base = DATASETS_ROOT
    cls_2d_tgt_dir = base / 'Classification' / 'AMD_DME_3D'
    cls_3d_tgt_dir = base / 'Classification_3D' / 'AMD_DME_3D'
    S2d = ClsSplit('classification_2d', 'AMD_DME_3D')
    S3d = ClsSplit('classification_3d', 'AMD_DME_3D')
    S = SegSplit('AMD_DME_3D')

    paths = {
        'AMD': path_dir / 'AMD_labeled',
        'DME': path_dir / 'DME_labeled',
    }
    # Seg info:
    #     AMD: 255: PED
    #     DME: 255: IRF -> 127
    for cls_name, c_path in paths.items():
        for vol_fn in sorted((c_path / 'input').iterdir()):
            r = S3d.get(f'{vol_fn.stem}.npz')
            if r is None:
                continue
            split_name, cls_name_json = r
            assert cls_name_json == cls_name
            seg_fn = c_path / 'label' / f'{vol_fn.stem}_mask.tif'
            assert seg_fn.exists(), seg_fn
            vol = io.imread(vol_fn)
            seg = io.imread(seg_fn)
            if cls_name == 'DME':
                seg[seg == 255] = 127
            cls_2d_fn = cls_2d_tgt_dir / split_name / cls_name / f'{vol_fn.stem}.png'
            cls_3d_fn = cls_3d_tgt_dir / split_name / cls_name / f'{vol_fn.stem}.npz'
            cls_2d_fn.parent.mkdir(parents=True, exist_ok=True)
            cls_3d_fn.parent.mkdir(parents=True, exist_ok=True)
            assert S2d.get(f'{vol_fn.stem}.png') == r
            io.imsave(cls_2d_fn, vol[vol.shape[0] // 2])
            np.savez_compressed(cls_3d_fn, vol=vol, seg=seg)
            for i in range(vol.shape[0]):
                name = f'{vol_fn.stem}_{i:03d}.png'
                for split in S.get(name):
                    save_seg_sample(S, split, name, vol[i], {'semseg': seg[i]})
    S.write_info(TGT_SEG_DIR / S.dataset)
    S2d.report()
    S3d.report()
    S.report()


def amd_sd_seg():
    path_dir = RAW_DATA_ROOT / 'AMD-SD/images'
    # NOTE: The splits are at the volume level (the original splits in the B-scan level were not used).
    S = SegSplit('AMD_SD_vol')
    mapping = {
        # SRF: red
        # IRF: green
        # PED: blue
        # IS/OS: magenta
        # SHRM: yellow
        (255, 0, 0): 50,  # SRF
        (0, 255, 0): 100,  # IRF
        (0, 0, 255): 150,  # PED
        (255, 0, 255): 200,  # IS/OS
        (255, 255, 0): 250,  # SHRM
        (255, 255, 255): 0,  # Weird borders after thresholding
    }
    for vol_fn in sorted(path_dir.iterdir()):
        for bs_fn in sorted(vol_fn.iterdir()):
            vol_id, bscan_id = bs_fn.stem.split('_')
            save_name = f'{int(vol_id):04d}_{int(bscan_id):03d}.png'
            splits = S.get(save_name)
            if not splits:
                continue
            img = io.imread(bs_fn)
            half_width = img.shape[1] // 2
            bscan = convert_to_grayscale(img[:, :half_width])
            seg = img[:, half_width:]
            # Binarize seg
            seg[seg > 127] = 255
            seg[seg <= 127] = 0
            new_seg = np.zeros(seg.shape[:2], dtype=np.uint8)
            for color, value in mapping.items():
                mask = np.all(seg == color, axis=-1)
                new_seg[mask] = value
            assert set(np.unique(new_seg)).issubset(set(mapping.values()))
            for split in splits:
                save_seg_sample(S, split, save_name, bscan, {'semseg': new_seg})
    S.write_info(TGT_SEG_DIR / S.dataset)
    S.report()


COLOR_TO_GRAY_OIMHS = {
    (0, 0, 0): 0,        # black: Background
    (0, 255, 0): 63,     # green: Retina
    (255, 255, 0): 126,  # yellow: Choroid
    (255, 0, 0): 189,    # red: Macular hole
    (0, 0, 255): 252,    # blue: IRF
}

def color_to_grayscale_oimhs(vol_rgb):
    """
    vol_rgb: [S, H, W, 3] uint8 array, each pixel one of the colors above.
    Returns: [S, H, W] uint8 grayscale volume.
    """
    S, H, W, _ = vol_rgb.shape
    gray = np.zeros((S, H, W), dtype=np.uint8)

    for rgb, gray_val in COLOR_TO_GRAY_OIMHS.items():
        mask = np.all(vol_rgb == np.array(rgb), axis=-1)  # [S, H, W]
        gray[mask] = gray_val

    return gray

def oimhs_seg():
    S = SegSplit('OIMHS')
    # The volumes are the ones of the 3D classification dataset (see oimhs())
    for split_3d, cls_name, vol_name in ClsSplit('classification_3d', 'OIMHS'):
        data = np.load(TGT_BASE_PATH / 'OIMHS' / split_3d / cls_name / vol_name)
        vol = data['vol']
        segg = color_to_grayscale_oimhs(data['seg'])
        cls_name_disp = cls_name.replace('MacularHole', 'MH')
        for i in range(vol.shape[0]):
            name = f'{cls_name_disp}_{Path(vol_name).stem}_{i:03d}.png'
            for split in S.get(name):
                save_seg_sample(S, split, name, vol[i], {'semseg': segg[i]})
    S.write_info(TGT_SEG_DIR / S.dataset)
    S.report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str, help='Function to execute')
    args = parser.parse_args()

    globals()[args.function]()
