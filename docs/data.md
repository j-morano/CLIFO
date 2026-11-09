# Data

The datasets are **not** distributed with this repository. Download them from their original sources, convert them with
the scripts in [`data_processing/`](../data_processing/) and use the static splits in [`splits/`](../splits/).

## Layout of the processed datasets

All paths are relative to `eval/_datasets/` (override with `CLIFO_DATASETS`).

```
Classification/<dataset>/<split>/<class>/<image>            # B-scan classification (and change detection: MARIO_T1)
Classification_3D/<dataset>/<split>/<class>/<volume>.npz    # volumes, array "vol" with shape (B-scans, H, W)
Segmentation/<dataset>/<split>/bscan/<name>.png             # B-scans
Segmentation/<dataset>/<split>/semseg/<name>.png            # masks (gray values defined in INFO.json)
Segmentation/<dataset>/<split>/semseg_2/<name>.png          # second annotation (UMN; RETOUCH test: semseg_Radboud)
Segmentation/<dataset>/INFO.json                            # label map of the masks
```

`splits/*.json` lists the exact files of every dataset, split and class (see [`splits/README.md`](../splits/README.md)).
The scripts in `data_processing/` read them to decide the split and class of every file (no random seeds), and print
how many files of the split were produced.

## Conversion of the original datasets

Set `RAW_DATA_ROOT` (original downloads) and `DATASETS_ROOT` (output), then run a function of a script, e.g.

```bash
export RAW_DATA_ROOT=/path/to/raw_datasets DATASETS_ROOT=$PWD/eval/_datasets
python data_processing/build_classification_2d.py octa500
python data_processing/build_volumes_and_segmentation.py octa500
python data_processing/build_2d_from_3d.py --splits splits/classification_2d.json \
    --root3d $DATASETS_ROOT/Classification_3D --out2d $DATASETS_ROOT/Classification
```

The scripts expect the original downloads in the following folders (names as used in the code):

| Script | Function | Expected raw data (under `RAW_DATA_ROOT`) |
|---|---|---|
| `build_classification_2d.py` | `kermany_correct` | `Kermany_correct` |
| `build_classification_2d.py` | `oct_c8` | `OCT-C8/RetinalOCT_Dataset/RetinalOCT_Dataset` |
| `build_classification_2d.py` | `oimhs` | `OIMHS_dataset/Demographics of the participants.xlsx`<br>`OIMHS_dataset/Images` |
| `build_classification_2d.py` | `octandeyefundus` | `OCT-AND-EYE-FUNDUS-DATASET-main` |
| `build_classification_2d.py` | `poagd` | `POAG_detection/glaucoma_detection` |
| `build_classification_2d.py` | `goals` | `FoundOPTIMA_Downstream/GOALS/` |
| `build_classification_2d.py` | `octmanualdelineations` | `OCT_Manual_Delineations-2018_June_29_b/OCT_Manual_Delineations-2018_June_29` |
| `build_classification_2d.py` | `banglaoct2025` | `BanglaOCT2025/BanglaOCT2025_Raw_130_Images_Only` |
| `build_classification_2d.py` | `octa500` | `OCTA-500` |
| `build_classification_2d.py` | `mmcamd` | `MMC-AMD` |
| `build_classification_2d.py` | `m3ret` | `M3Ret` |
| `build_classification_2d.py` | `octave` | `OCTAVE/nnUNet_raw/Dataset001_OCTAVE/imagesTr` |
| `build_classification_2d.py` | `mario` | `MARIO/Task_1` |
| `build_volumes_and_segmentation.py` | `oimhs` | `OIMHS_dataset/Images` |
| `build_volumes_and_segmentation.py` | `poagd` | `POAG_detection/glaucoma_detection` |
| `build_volumes_and_segmentation.py` | `umn` | `FoundOPTIMA_Downstream/UMN` |
| `build_volumes_and_segmentation.py` | `harvardglaucoma` | `FoundOPTIMA_Downstream/Harvard_Glaucoma_Detection/Harvard Glaucoma Detection and Progression with 1000 Samples (Harvard-GDP1000)/Bscan` |
| `build_volumes_and_segmentation.py` | `dukes` | `FoundOPTIMA_Downstream/Duke_Srinivasan/Publication_Dataset` |
| `build_volumes_and_segmentation.py` | `noor` | `FoundOPTIMA_Downstream/Noor_Eye_Hospital/Data` |
| `build_volumes_and_segmentation.py` | `octmanualdelineations` | `OCT_Manual_Delineations-2018_June_29_b/OCT_Manual_Delineations-2018_June_29` |
| `build_volumes_and_segmentation.py` | `banglaoct2025` | `BanglaOCT2025/BanglaOCT2025_Raw_130_Images_Only` |
| `build_volumes_and_segmentation.py` | `octa500` | `OCTA-500` |
| `build_volumes_and_segmentation.py` | `m3ret` | `M3Ret` |
| `build_volumes_and_segmentation.py` | `olives_baseline` | `FoundOPTIMA_Downstream/OLIVES/OLIVES` |
| `build_volumes_and_segmentation.py` | `gamma` | `GAMMA/Glaucoma_grading/training/multi-modality_images` |
| `build_volumes_and_segmentation.py` | `octave` | `OCTAVE/nnUNet_raw/Dataset001_OCTAVE/imagesTr` |
| `build_volumes_and_segmentation.py` | `rvo_lesion_seg` | `RVO-Lesion/Image_Seg` |
| `build_volumes_and_segmentation.py` | `retouch_test_seg` | `RETOUCH` |
| `build_volumes_and_segmentation.py` | `amd_dme_cls_seg` | `AMD_DME_3D_Dataset` |
| `build_volumes_and_segmentation.py` | `amd_sd_seg` | `AMD-SD/images` |
| `build_volumes_and_segmentation.py` | `octave_seg` | 3D `OCTAVE` (from `octave`) |
| `build_volumes_and_segmentation.py` | `oimhs_seg` | 3D `OIMHS` (from `oimhs`) |
| `build_volumes_and_segmentation.py` | `umn_seg` | 3D `UMN` (from `umn`) |

Not covered by the scripts (see `data_processing/README.md`): the cross-dataset sets, 2D Duke iAMD, OCTDL and OCTID (the
original images, split as in `splits/`), AROI / Duke DME / Duke iAMD / GOALS segmentation (except the sclera removal) and
the RETOUCH train/validation sets. These are fully described by the split files.
