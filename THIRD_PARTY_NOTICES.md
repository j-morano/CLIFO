# Third-party code

This repository includes code adapted from the following projects. Each file keeps its original copyright header,
where it had one. The licenses below are the ones of the upstream repositories.

| Code | Files | Upstream | License |
|---|---|---|---|
| MultiMAE (EPFL VILAB) | `multimae/` | https://github.com/EPFL-VILAB/MultiMAE | CC BY-NC 4.0 |
| MAE (Meta) and RETFound | `eval/mutils/vit.py`, `eval/mutils/pos_embed.py`, `eval/models/urfound/` (MAE-based) | https://github.com/facebookresearch/mae, https://github.com/rmaphoh/RETFound_MAE | CC BY-NC 4.0 |
| OCTCube (OCTCube-M group; contains code by Meta) | `eval/models/octcube/` | https://github.com/ZucksLiu/OCTCubeM | BSD 2-Clause (see file headers) |
| timm (Ross Wightman) | `utils/model_ema.py`, `utils/registry.py`, `utils/model_builder.py` | https://github.com/huggingface/pytorch-image-models | Apache 2.0 |
| DINO (Meta) | parts of `run_pretraining_bscan.py` | https://github.com/facebookresearch/dino | Apache 2.0 |
| Asymmetric loss | `AsymmetricLoss` in `run_pretraining_bscan.py` | https://github.com/Alibaba-MIIL/ASL | MIT |
| ConvNeXt (Meta) | `eval/seg_heads.py` | https://github.com/facebookresearch/ConvNeXt | MIT |
| MONAI | `eval/mutils/gdice.py` | https://github.com/Project-MONAI/MONAI | Apache 2.0 |

The pretrained weights of the external models used as baselines (RETFound, VisionFM, OCTCube, UrFound, MIRAGE) are not
distributed here and are subject to the licenses of their authors.

## Copyright notices and modifications

The following notices are given here instead of in the headers of the files. All the files listed below were modified
for CLIFO (new inputs and outputs for OCT B-scans, layer maps and lesion-presence labels, losses, pipelines).

* **MultiMAE** (https://github.com/EPFL-VILAB/MultiMAE, CC BY-NC 4.0)
  "Copyright (c) EPFL VILAB. All rights reserved."
  Adapted in: `multimae/` (including `criterion.py`, `parts/input_adapters.py`, `parts/output_adapter_utils.py`),
  `utils/data_constants.py`.
* **MAE** (https://github.com/facebookresearch/mae, CC BY-NC 4.0) and **RETFound** (https://github.com/rmaphoh/RETFound_MAE, CC BY-NC 4.0)
  "Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved."
  Adapted in: `eval/mutils/vit.py`, `eval/mutils/pos_embed.py`.
* Files that credit timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv or ConvNeXt in their headers keep those attribution lines.

No code of **VisionFM** (https://github.com/ABILab-CUHK/VisionFM) is included. `eval/fm_config.py` only uses its published
normalization constants and loads its released weights into a standard ViT.

The third-party code in `eval/models/octcube/` and `eval/models/urfound/` is included as distributed by its authors (OCTCube
reduced to the files used here) and keeps its original headers and licenses.
