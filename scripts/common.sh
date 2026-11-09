# Shared settings of the evaluation scripts (sourced, not executed).
#
# Environment variables (all optional):
#   CLIFO_OUTPUT    root of the outputs (features, probing, segmentation). Default: eval/__output
#   CLIFO_DATASETS  root of the processed datasets. Default: eval/_datasets
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/eval"
export PYTHONPATH="$ROOT:$PWD:${PYTHONPATH:-}"
export CLIFO_OUTPUT="${CLIFO_OUTPUT:-$PWD/__output}"

SEEDS=(0 1 2 3 4 5 6)
SEEDS_CHD=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)

DATASETS_2D=(
    AMD_DME_3D BanglaOCT2025 Duke_iAMD GAMMA GOALS Harvard_Glaucoma Kermany_correct M3Ret MMC_AMD Noor_Eye_Hospital
    Noor_to_UMNandDukeSrinivasan_cross OAEFD_DME OAEFD_DR OCTA500 OCTAVE OCTDL OCTID OCT_C8 OCT_MD_MS OIMHS
    OLIVES_baseline POAGD UMNandDukeSrinivasan_to_Noor_cross
)
DATASETS_3D=(
    AMD_DME_3D BanglaOCT2025 GAMMA Harvard_Glaucoma M3Ret Noor_Eye_Hospital Noor_to_UMNandDukeSrinivasan_cross OCTA500
    OCTAVE OCT_MD_MS OIMHS OLIVES_baseline POAGD UMNandDukeSrinivasan_to_Noor_cross
)
DATASETS_SEG=(
    AMD_DME_3D AMD_SD_vol AROI_nosclera Duke_DME_nosclera GOALS OCTAVE OIMHS RETOUCH RVO_Lesion UMN
)

# "<model key in fm_config.py> <pooling>"
MODELS_2D=(
    "octcube cls_patch" "retfound patch" "visionfm-base cls_patch" "urfound cls_patch"
    "miragev1-large patch" "CLIFO patch_lesion"
)
MODELS_CHD=(
    "octcube-chd cls_patch" "retfound-chd patch" "visionfm-base-chd cls_patch" "urfound-chd cls_patch"
    "miragev1-large-chd patch" "CLIFO_ChD patch_lesion"
)
MODELS_3D=("octcube3d_60 cls_patch" "CLIFO3D_60 2avg" "CLIFO3D 2avg")
MODELS_SEG=(CLIFO visionfm-base miragev1-large octcube retfound urfound)
