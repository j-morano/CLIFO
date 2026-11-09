"""Qualitative segmentation figures: colour-coded predictions of every model for the given test B-scans.

Colours: grey level of the class = correct, red = missed structure (false negative), cyan = predicted structure where
there is none (false positive), violet = structure present in both, but with a different class.

Usage (from eval/):
    python qualitative_seg_results.py OCTAVE/<file>.png GOALS/<file>.png
Each image is given as <dataset>/<file name> (a file of <dataset>/test/semseg). The figures are saved in
GEN_RESULTS/qualitative_seg/<dataset>_<file name without extension>/ (bscan.png, ground_truth.png and one <model>.png
per model).
"""
from pathlib import Path
import argparse

import numpy as np
from skimage import io
from skimage.transform import resize
from paths import OUTPUT, DATASETS, GEN_RESULTS


RES_PATH = OUTPUT / 'seg' / 'SOTA_comparison_new' / '0'
DATA_PATH = DATASETS / 'Segmentation'
SAVE_DIR = GEN_RESULTS / 'qualitative_seg'


models = {
    "miragev1-large_convnext_w": "MIRAGE",
    "octcube_convnext_w": "OCTCube",
    "CLIFO_convnext_w": "CLIFO",
    "retfound_convnext_w": "RETFound",
    "urfound_convnext_w": "UrFound",
    "visionfm-base_convnext_w": "VisionFM",
}


# Classes of the AROI model (trained with the AROI labels) -> classes of Duke iAMD
mapping_aroi_dukeiamd = {
    0: 51,
    23: 102,
    46: 102,
    69: 153,
    92: 204,
    # Map to invalid
    115: 0,
    138: 0,
    161: 0,
}


def colorize_prediction(gt, pred, bscan, remap=False):
    """Colour-codes a prediction against the ground truth. Returns the RGB image, the normalized ground truth and the
    B-scan (both cropped when remap=True). remap=True is for the AROI -> Duke iAMD evaluation."""
    # Resize pred to match gt if needed
    if pred.shape != gt.shape:
        pred = resize(
            pred,
            gt.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)

    if remap:
        nonzero_counts = np.sum(gt > 0, axis=0)
        min_pixels = 5
        nonzero_cols = np.where(nonzero_counts > min_pixels)[0]
        if len(nonzero_cols) > 0:
            col_start = nonzero_cols[0]
            col_end = nonzero_cols[-1] + 1
            gt = gt[:, col_start:col_end]
            pred = pred[:, col_start:col_end]
            bscan = bscan[:, col_start:col_end]
        pred = np.vectorize(mapping_aroi_dukeiamd.get)(pred).astype(np.uint8)
        pred[pred == 51] = 0
        gt[gt == 51] = 0

    # Allocate rgb AFTER any cropping, using the (possibly new) shape
    H, W = gt.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Make brighter: move all intensities > 0 to the range 100-255.
    mapping = {0: 0}
    for i in range(1, 256):
        mapping[i] = int(100 + (i / 255) * (255 - 100))
    gt = np.vectorize(mapping.get)(gt).astype(np.uint8)
    pred = np.vectorize(mapping.get)(pred).astype(np.uint8)

    gt_bg = (gt == 0)
    pred_bg = (pred == 0)
    correct = (gt == pred) & ~gt_bg  # true positive, correct class
    false_neg = (~gt_bg) & pred_bg   # GT has structure, pred says background
    false_pos = gt_bg & (~pred_bg)   # GT background, pred says structure
    wrong_class = (~gt_bg) & (~pred_bg) & (gt != pred)  # both non-bg, different classes

    # True positives: grayscale value = class index (using gt's class value)
    rgb[correct] = np.stack([gt[correct]] * 3, axis=-1)

    # False negatives (missed structure): red
    rgb[false_neg] = [255, 0, 0]

    # False positives (predicted structure where none exists): cyan
    rgb[false_pos] = [0, 255, 255]

    # Wrong class (structure present in both, but mismatched class): violet
    rgb[wrong_class] = [148, 0, 211]

    return rgb, gt, bscan


def make_figure(dataset, file_name):
    """Saves the ground truth, the B-scan and the colour-coded prediction of every model for one test B-scan."""
    if "Duke_iAMD" in dataset:
        gt_path = DATA_PATH / dataset / "semseg"
        remap = True
    else:
        gt_path = DATA_PATH / dataset / "test" / "semseg"
        remap = False

    gt_file = gt_path / file_name
    assert gt_file.exists(), gt_file
    gt = io.imread(gt_file)
    bscan = io.imread(gt_file.parent.parent / "bscan" / file_name)

    c_save_dir = SAVE_DIR / f"{dataset}_{Path(file_name).stem}"
    c_save_dir.mkdir(parents=True, exist_ok=True)

    gt_saved = False
    for model in models:
        if "Duke_iAMD" in dataset:
            pred_path = RES_PATH / "AROI_nosclera" / model / "Duke_iAMD_labeled_nosclera_70_predictions"
        else:
            pred_path = RES_PATH / dataset / model / "test_predictions"
        pred_file = pred_path / file_name
        assert pred_file.exists(), pred_file
        color_pred, gt_norm, bscan_norm = colorize_prediction(gt, io.imread(pred_file), bscan, remap=remap)
        if not gt_saved:
            io.imsave(c_save_dir / "ground_truth.png", gt_norm, check_contrast=False)
            io.imsave(c_save_dir / "bscan.png", bscan_norm)
            gt_saved = True
        io.imsave(c_save_dir / f"{models[model]}.png", color_pred, check_contrast=False)
    print(f"Saved to {c_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="B-scans as <dataset>/<file name>, e.g. OCTAVE/CNV_10_050.png")
    args = parser.parse_args()
    for image in args.images:
        dataset, file_name = image.split("/")
        make_figure(dataset, file_name)
