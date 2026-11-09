# Dataset splits

Static splits for all the downstream datasets used in the paper. The datasets themselves are **not** included:
download them from their original sources and convert them with the scripts in `data_processing/`.

| File | Content |
|---|---|
| `classification_2d.json` | dataset -> split -> class -> image file names (B-scan classification) |
| `classification_3d.json` | dataset -> split -> class -> volume file names (`.npz`, key `vol`); also used for retrieval |
| `change_detection.json` | MARIO (change detection): two B-scans stacked in the image channels |
| `segmentation.json` | dataset -> label map (`INFO.json`) and, per split, the file names of `bscan/`, `semseg/`, ... |

The file names are the ones produced by the scripts in `data_processing/`, which read these files to decide the split
and class of every file.
