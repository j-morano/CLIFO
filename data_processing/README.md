# Data processing (downstream datasets)

First version of the code used to convert the original downstream datasets into the format used by the evaluation
code (see `../splits/` for the exact splits). **Work in progress.**

* `build_classification_2d.py`: 2D classification datasets (B-scans, mostly the central B-scan of each volume) and change detection (MARIO).
* `build_volumes_and_segmentation.py`: 3D classification datasets (`.npz` with key `vol`) and segmentation datasets (B-scans + masks).
* `build_2d_from_3d.py`: 2D images of GAMMA, Harvard Glaucoma, Noor Eye Hospital and OLIVES, taken from the 3D volumes.
* `remove_sclera.py`: removal of the sclera label in the AROI and Duke segmentation datasets.
* `img_utils.py`: helper functions.

* `split_utils.py`: reads `../splits/*.json`.

Usage: `RAW_DATA_ROOT=<raw datasets> DATASETS_ROOT=<output> python <script>.py <function_name>`.

The builders are driven by the static splits in `../splits/*.json`: each function walks the original dataset (or the
file list in the JSON), takes the split and class of every file from the JSON, and prints how many files of the split
were produced (a warning lists those that were not). No random seeds are involved, except for the noise of `mmcamd()`
(fixed seed).

Known limitations of this version:
* Paths are hard-coded relative to `RAW_DATA_ROOT` (folder names of the original downloads) and may need adapting.
* The 3D datasets are built directly from the original data, the 2D ones of GAMMA, Harvard Glaucoma, Noor Eye Hospital
  and OLIVES from the 3D ones (`build_2d_from_3d.py`). `octave_seg()`, `oimhs_seg()` and `umn_seg()` need the 3D
  datasets (`octave()`, `oimhs()`, `umn()`).
* `amd_dme_cls_seg()` builds the 2D, 3D and segmentation versions of AMD_DME_3D together.
* No code is included for: the cross-dataset sets (UMN + Duke Srinivasan <-> Noor Eye Hospital), 2D Duke_iAMD, the AROI,
  Duke DME, Duke iAMD and GOALS segmentation sets (except the sclera removal), or the RETOUCH train/val sets.
  OCTDL and OCTID are the original files, only split into train/val/test.
* `dukes()` and `umn()` do not use the splits (they convert the complete datasets).
* Not tested on the original data (only on synthetic data).
