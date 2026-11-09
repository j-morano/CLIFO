import os
from pathlib import Path
import argparse
import shutil

from skimage import io
import numpy as np
# package for loading .mat
from scipy.ndimage import label
import pandas as pd

from img_utils import remove_white_borders
from split_utils import ClsSplit

RAW_DATA_ROOT = Path(os.environ.get('RAW_DATA_ROOT', '/path/to/raw_datasets'))  # downloads of the original datasets
DATASETS_ROOT = Path(os.environ.get('DATASETS_ROOT', './_datasets'))  # processed datasets (output)


BASE_DATA_PATH = RAW_DATA_ROOT


TGT_BASE_PATH = DATASETS_ROOT / 'Classification'


def convert_to_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to grayscale using luminosity method
        return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype('uint8')
    elif len(image.shape) == 2:
        # Image is already grayscale
        return image
    else:
        raise ValueError("Unsupported image format")


'''NOTE:
For very large datasets like Kermany and C8, the idea for linear probing is the following: keep the test set as it is,
and split the original validation set into train/val (images ordered by name, the first 60% for training and the rest
for validation). In addition, both datasets are cleaned by removing the white borders. The final splits are in
../splits/classification_2d.json.
'''


def process_kermany_img(img_fn, tgt_path, cls_name, split):
    img = io.imread(img_fn)
    img = convert_to_grayscale(img)
    img = remove_white_borders(img)
    tgt_fn = tgt_path / split / cls_name / img_fn.name
    tgt_fn.parent.mkdir(parents=True, exist_ok=True)
    print(' ', tgt_fn)
    io.imsave(tgt_fn, img)


def process_kermany_like_dataset(tgt_path, path_dir, dataset):
    # The splits come from the JSON: the original test set is the test set, and the original validation set is split
    #   into train/val.
    S = ClsSplit('classification_2d', dataset)
    for subset in ['test', 'val']:
        for disease in (path_dir / subset).iterdir():
            for img_fn in disease.iterdir():
                r = S.get(img_fn.name)
                if r is not None:
                    process_kermany_img(img_fn, tgt_path, r[1], r[0])
    S.report()


def kermany_correct():
    path_dir = BASE_DATA_PATH / 'Kermany_correct'
    tgt_path = TGT_BASE_PATH / 'Kermany_correct'
    process_kermany_like_dataset(tgt_path, path_dir, 'Kermany_correct')


def oct_c8():
    path_dir = BASE_DATA_PATH / 'OCT-C8/RetinalOCT_Dataset/RetinalOCT_Dataset'
    tgt_path = TGT_BASE_PATH / 'OCT_C8'
    process_kermany_like_dataset(tgt_path, path_dir, 'OCT_C8')


def oimhs():
    path_dir = BASE_DATA_PATH / 'OIMHS_dataset/Images'
    tgt_path = TGT_BASE_PATH / 'OIMHS'
    # File names: <patient_id>_<eye_id>_<index>.png, where <index> is the B-scan of the eye (volume) with the largest
    #   macular hole. The classes are the stage of the macular hole (1+2+3 and 4). The image has the B-scan in the left
    #   half and the segmentation in the right half.
    S = ClsSplit('classification_2d', 'OIMHS')
    for split, cls_name, fn in S:
        patient_id, eye_id, idx = Path(fn).stem.split('_')
        img = io.imread(path_dir / eye_id / f'{idx}.png')
        left_half = img[:, :img.shape[1] // 2]
        tgt_fn = tgt_path / split / cls_name / fn
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        print(' ', tgt_fn)
        io.imsave(tgt_fn, left_half)
    S.report()


def octandeyefundus():
    '''
    One dataset for each label type: DME (DME present or not) and DR (no DR, NPDR, PDR). The labels are in OCT.csv of the
    original dataset (columns Name, DME, DR); here the class of each image comes from the splits. Images with unknown
    DR label ('-') are only in the DME dataset.
    For example: '1221_OD_o_3' corresponds to the 3rd sample picture of OCT image from the right eyeball of subject
    number 1221.
    '''
    path_dir = BASE_DATA_PATH / 'OCT-AND-EYE-FUNDUS-DATASET-main'
    tgt_paths = {'OAEFD_DME': TGT_BASE_PATH / 'OAEFD_DME', 'OAEFD_DR': TGT_BASE_PATH / 'OAEFD_DR'}
    splits = {name: ClsSplit('classification_2d', name) for name in tgt_paths}
    for img_fn in (path_dir / 'OCT').glob('*/*.*'):
        for name, S in splits.items():
            r = S.get(img_fn.name)
            if r is None:
                continue
            tgt_fn = tgt_paths[name] / r[0] / r[1] / img_fn.name
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_fn, tgt_fn)
    for S in splits.values():
        S.report()


def poagd():
    path_dir = BASE_DATA_PATH / 'POAG_detection/glaucoma_detection'
    tgt_base = TGT_BASE_PATH / 'POAGD'
    S = ClsSplit('classification_2d', 'POAGD')
    for img_fn in sorted(path_dir.glob('*.npy')):
        r = S.get(img_fn.name.replace('.npy', '.png'))
        if r is None:
            continue
        volume = np.load(img_fn)
        bscan = volume[volume.shape[0] // 2]
        tgt_fn = tgt_base / r[0] / r[1] / img_fn.name.replace('.npy', '.png')
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(tgt_fn, bscan)
    S.report()


def goals():
    # The classes (GC_Label in GC_GT.csv: 0 Normal, 1 Glaucoma) come from the splits
    path_dir = BASE_DATA_PATH / 'FoundOPTIMA_Downstream/GOALS/'
    tgt_base = TGT_BASE_PATH / 'GOALS'
    S = ClsSplit('classification_2d', 'GOALS')
    for img_fn in sorted((path_dir / 'Image').glob('*.*')):
        r = S.get(img_fn.name)
        if r is None:
            continue
        tgt_fn = tgt_base / r[0] / r[1] / img_fn.name
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_fn, tgt_fn)
    S.report()


def octmanualdelineations():
    import eyepy

    # https://iacl.ece.jhu.edu/index.php?title=Resources
    path = RAW_DATA_ROOT / 'OCT_Manual_Delineations-2018_June_29_b/OCT_Manual_Delineations-2018_June_29'
    S = ClsSplit('classification_2d', 'OCT_MD_MS')
    # ID,Age,Gender,Eye,Diagnosis
    for img_fn in sorted((path / 'vol').iterdir()):
        id_ = img_fn.stem.split('_')[0]
        r = S.get(f'{id_}.png')
        if r is None:
            continue
        vol = eyepy.import_heyex_vol(img_fn).data
        bscan = vol[vol.shape[0] // 2]
        tgt_fn = TGT_BASE_PATH / 'OCT_MD_MS' / r[0] / r[1] / f'{id_}.png'
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(tgt_fn, bscan)
    S.report()


def banglaoct2025():
    path_dir = RAW_DATA_ROOT / 'BanglaOCT2025/BanglaOCT2025_Raw_130_Images_Only'
    # NOTE: Very few images for validation and test in the original split. The splits were recreated (at patient level).
    S = ClsSplit('classification_2d', 'BanglaOCT2025')
    for disease in sorted(path_dir.iterdir()):
        if not disease.is_dir():
            continue
        for patient in sorted(disease.iterdir()):
            if not patient.is_dir():
                continue
            bscan_fns = sorted(patient.glob('oct_c_*.bmp'))
            if not bscan_fns:
                print(f'No B-scans found for {patient}')
                continue
            middle_bscan_fn = bscan_fns[len(bscan_fns) // 2]
            fn = f'{patient.name}_{middle_bscan_fn.stem}.png'
            r = S.get(fn)
            if r is None:
                continue
            bscan_gray = convert_to_grayscale(io.imread(middle_bscan_fn))
            tgt_fn = TGT_BASE_PATH / 'BanglaOCT2025' / r[0] / r[1] / fn
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            print(' ', tgt_fn)
            io.imsave(tgt_fn, bscan_gray)
    S.report()


def octa500():
    path_dir = RAW_DATA_ROOT / 'OCTA-500'
    # NOTE: Very few cases in total. The classes are Normal (NORMAL, 160 volumes) and Diseased (CNV, DR and AMD,
    #   40 volumes). The disease of each volume is in 'Text labels.xlsx' of the original dataset.
    S = ClsSplit('classification_2d', 'OCTA500')
    data_dir = path_dir / 'dataset_3m' / 'OCT'
    for vol_dir in sorted(data_dir.iterdir()):
        if not vol_dir.is_dir():
            continue
        indices = sorted([int(fn.stem) for fn in vol_dir.glob('*.bmp')])
        middle_idx = indices[len(indices) // 2]
        fn = f'{vol_dir.name}_{middle_idx}.png'
        r = S.get(fn)
        if r is None:
            continue
        middle_bscan_gray = convert_to_grayscale(io.imread(vol_dir / f'{middle_idx}.bmp'))
        tgt_fn = TGT_BASE_PATH / 'OCTA500' / r[0] / r[1] / fn
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        print(' ', tgt_fn)
        io.imsave(tgt_fn, middle_bscan_gray)
    S.report()


def mmcamd():
    # https://github.com/search?q=repo%3Ali-xirong%2Fmmc-amd%20dryAMD&type=code
    path_dir = BASE_DATA_PATH / 'MMC-AMD'
    oct_dir = path_dir / 'ImageData' / 'oct'
    S = ClsSplit('classification_2d', 'MMC_AMD')
    rng = np.random.default_rng(0)
    for oct_fn in sorted(oct_dir.glob('*.jpg')):
        r = S.get(oct_fn.name)
        if r is None:
            continue
        subset, disease = r
        oct = io.imread(oct_fn)
        # Add random noise L in the bottom left corner to remove some
        #   identifying info from the image (e.g. scale bar)
        oct = convert_to_grayscale(oct)
        # Copy the crop next to the bottom left corner and add some
        #   noise. NOTE: do this to make sure all images from
        #   diffferent classes look the same, as some classes
        #   always had this thing and others not.
        noise = rng.integers(0, 5, size=(64, 48), dtype=np.uint8)
        center_w = oct.shape[1] // 2
        crop = oct[-64:, center_w - 24:center_w + 24 ]
        oct[-64:, :48] = crop + noise
        tgt_fn = TGT_BASE_PATH / 'MMC_AMD' / subset / disease / oct_fn.name
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        print(' ', tgt_fn)
        io.imsave(tgt_fn, oct)
    S.report()


def m3ret():
    path_dir = RAW_DATA_ROOT / 'M3Ret'
    info_fn = path_dir / 'labels_data_split/oct-macular-v4.6.xlsx'
    # The splits and the classes (ME, ME_DR, DR, Neither) come from the JSON. The excel gives the patient name, needed
    #   to find the folder of each volume.
    S = ClsSplit('classification_2d', 'M3Ret')
    for sheet in ['train-1216', 'val-406', 'test-406']:
        info = pd.read_excel(info_fn, sheet_name=sheet)
        # Index(['patient ID', 'patient name', 'laterality', 'year', 'month', 'day',
        #        'ME', 'DR', 'glaucoma', 'cataract', 'optic_atrophy', 'birthday', 'age',
        #        'gender', 'Images'],
        #       dtype='str')
        for idx, row in info.iterrows():
            pat_id = row['patient ID']
            laterality = row['laterality']
            name = row['patient name']
            img_dir = path_dir / 'OCTImages-Macular-v3.4' / f'{str(pat_id).zfill(10)}+{name}' / laterality / 'study1'
            assert img_dir.exists(), f'Image directory {img_dir} does not exist'
            img_fns = sorted(img_dir.glob('volume_1_*.png'))
            bscan_ids = sorted([int(fn.stem.split('_')[-1]) for fn in img_fns])
            middle = bscan_ids[len(bscan_ids) // 2]
            middle_fn = img_dir / f'volume_1_{middle}.png'
            assert middle_fn.exists(), f'Middle B-scan {middle_fn} does not exist'
            fn = f'{pat_id}_{laterality}_{middle_fn.stem}.png'
            r = S.get(fn)
            if r is None:
                continue
            tgt_fn = TGT_BASE_PATH / 'M3Ret' / r[0] / r[1] / fn
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            print(' ', tgt_fn)
            shutil.copy(middle_fn, tgt_fn)
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


def octave():
    path_dir = RAW_DATA_ROOT / 'OCTAVE/nnUNet_raw/Dataset001_OCTAVE/imagesTr'
    # File names: <disease>_<id>_<index of the central B-scan>.png
    S = ClsSplit('classification_2d', 'OCTAVE')
    for split, disease, fn in S:
        _, iid, middle_str = Path(fn).stem.split('_')
        img = io.imread(path_dir / f'OCTAVE_{disease}_{iid}_0000.tif')
        # Take the middle B-scan
        middle = img.shape[0] // 2
        assert int(middle_str) == middle
        bscan = convert_to_grayscale(img[middle])
        bscan = remove_white_part(bscan)
        tgt_fn = TGT_BASE_PATH / 'OCTAVE' / split / disease / fn
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        print(' ', tgt_fn)
        io.imsave(tgt_fn, bscan)
    S.report()


def mario():
    path_dir = RAW_DATA_ROOT / 'MARIO/Task_1'
    tgt_base = TGT_BASE_PATH / 'MARIO_T1'
    pattern_fn = 'df_task1_{subset}.csv'
    # The splits and classes (reduced, stable, worsened, other) come from the JSON
    S = ClsSplit('change_detection', 'MARIO_T1')
    for subset in ['train', 'val', 'test']:
        info_fn = path_dir / pattern_fn.format(subset=subset)
        info = pd.read_csv(info_fn)
        img_dir = path_dir / subset
        # id_patient,side_eye,BScan,image_at_ti,image_at_ti+1,label,split_type,LOCALIZER_at_ti+1,LOCALIZER_at_ti,sex,age_at_ti+1,age_at_ti,num_current_visit_at_i+1,num_current_visit_at_i,delta_t,case
        for idx, row in info.iterrows():
            img_1_fn = Path(row['image_at_ti'])
            img_2_fn = Path(row['image_at_ti+1'])
            fn = f'{img_1_fn.stem}_{img_2_fn.stem}.png'
            r = S.get(fn)
            if r is None:
                continue
            img_fn_1 = img_dir / img_1_fn
            img_fn_2 = img_dir / img_2_fn
            assert img_fn_1.exists(), img_fn_1
            assert img_fn_2.exists(), img_fn_2
            img_1 = convert_to_grayscale(io.imread(img_fn_1))
            img_2 = convert_to_grayscale(io.imread(img_fn_2))
            padding = np.zeros_like(img_1)
            img_combined = np.stack([img_1, img_2, padding], axis=-1)
            tgt_fn = tgt_base / r[0] / r[1] / fn
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(tgt_fn, img_combined)
            print(' ', tgt_fn)
    S.report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str, help='Function to execute')
    args = parser.parse_args()

    globals()[args.function]()
