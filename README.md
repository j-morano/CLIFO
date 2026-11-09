# CLIFO: CLInically guided Foundation model for retinal OCT

Code of the paper *CLIFO: …* (preprint / publication details to be added).

CLIFO is a foundation model for retinal optical coherence tomography (OCT) trained in two stages that follow the
hierarchy of OCT data: a **B-scan encoder** (ViT-B/32, 512×512 input) trained with masked image modelling plus
dense (layer maps) and global (lesion presence) pseudo-label supervision, and a **volume feature encoder (VFE)** trained
with a vision-language objective on clinical reports. This repository contains the code for both pretraining stages and
for the downstream evaluation (B-scan and volume diagnosis, change detection, retrieval, segmentation), the splits of all
the downstream datasets, and the code to generate the tables and figures of the paper.

> **Not released:** the pretraining data (private clinical data) and the code that generates the pseudo-labels and
> the reports; the datasets and the weights of the external models (see below).

## Repository layout

| Path | Content |
|---|---|
| `run_pretraining_bscan.py`, `multimae/`, `utils/`, `config.py`, `cfgs/pretrain/clifo_stage1.yaml` | Stage 1: B-scan encoder |
| `run_pretraining_vfe.py`, `docs/stage2_reports.md` | Stage 2: volume feature encoder (VFE) and report generation |
| `eval/` | Evaluation: feature extraction, linear probing, retrieval, segmentation, result gathering and plotting |
| `eval/fm_config.py` | Registry of the evaluated models: `CLIFO`, `CLIFO3D`, `CLIFO3D_60`, `miragev1-large`, `retfound`, `visionfm-base`, `octcube`, `octcube3d_60`, `urfound` (+ change-detection variants) |
| `splits/` | Static splits (JSON) of all the downstream datasets |
| `data_processing/` | Conversion of the original datasets (first version, see its README) |
| `scripts/` | Commands to reproduce the experiments |
| `inference/` | Minimal example to extract features with CLIFO |
| `docs/` | Data and stage-2 documentation |

## Setup

The project uses [uv](https://docs.astral.sh/uv/) (Python 3.12, PyTorch 2.11 with CUDA 12.6 wheels).

```bash
uv sync                      # core dependencies
uv sync --extra octcube      # FlashAttention, needed for the OCTCube baseline (compiles against the installed torch)
uv sync --extra report       # BLEU/ROUGE for the report-generation evaluation
uv sync --extra data         # packages needed by data_processing/
source .venv/bin/activate
```

## Weights

Put the weights in `_weights/` and link them for the evaluation code: `ln -s ../_weights eval/_weights`.

| File | Used by | Source |
|---|---|---|
| `CLIFO.pth` (B-scan encoder), `CLIFO_VFE.pt` (volume feature encoder) | CLIFO | Releases page of this repository |
| `MIRAGE-Base.pth` | Initialisation of stage 1 | [MIRAGE](https://github.com/j-morano/MIRAGE) |
| `MIRAGEv1-Large.pth` | MIRAGE baseline | [MIRAGE](https://github.com/j-morano/MIRAGE) |
| `RETFound-Large.pth` | RETFound baseline | [RETFound](https://github.com/rmaphoh/RETFound_MAE) |
| `VisionFM-Base.pth` | VisionFM baseline | [VisionFM](https://github.com/ABILab-CUHK/VisionFM) |
| `OCTCube.pth` | OCTCube baseline | [OCTCube](https://github.com/ZucksLiu/OCTCubeM) |
| `urfound_mm.pth` | UrFound baseline | UrFound (Yu et al., 2024) |

## Data

The datasets are not distributed. The exact splits are in [`splits/`](splits/) and the conversion of the original
datasets is described in [`docs/data.md`](docs/data.md). Paths are configured with environment variables
(`CLIFO_DATASETS`, `CLIFO_OUTPUT`, `CLIFO_RAW_RESULTS`, `CLIFO_GEN_RESULTS`; see `eval/paths.py`).

## Usage

Pretraining (the pretraining data is not released; the scripts document the settings of the paper):

```bash
scripts/pretrain_stage1.sh      # B-scan encoder
scripts/pretrain_stage2.sh      # volume feature encoder (see docs/stage2_reports.md)
```

Downstream evaluation, in this order:

```bash
scripts/extract_features.sh 2d  # also: 3d, chd. Caches the frozen features of every model and dataset
scripts/probe.sh 2d             # linear probing, 7 seeds (14 for chd)
scripts/retrieval.sh            # retrieval, t-SNE and class dendrograms (3D features)
scripts/data_efficiency.sh      # labelled samples per class, inference time and memory
scripts/segmentation.sh         # segmentation heads, evaluation and gathering
scripts/make_results.sh         # tables and figures
```

Extract the features of a single model on a single dataset:

```bash
cd eval && PYTHONPATH=. python -m run_cls_tuning --version SOTA_comparison_features --seed 0 --weights CLIFO \
    --pool patch_lesion --data_root _datasets/Classification/ --data_set GAMMA --smoothing 0.0 --balance \
    --device cuda --cache_features
```

## Status

First version of the code for the submission. The scripts have not been tested on a clean machine yet; paths and
instructions will be refined.

## License

The code of this repository is released under [CC BY-NC 4.0](LICENSE). It includes third-party code under its own
license, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
