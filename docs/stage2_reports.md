# Stage 2: report data

Stage 2 (`scripts/pretrain_stage2.sh`) trains the volume feature encoder (VFE) on the clinical reports of the OCT
volumes. The report data is not released (private clinical data). To train on your own data, provide the following.

## Reports: `cfgs/reports_clean_split.json`

A JSON file with a `train` and a `val` split. Each split maps a volume ID to its report. Only the field
`oct_findings` (the report text, the generation target) is required; the other fields are optional metadata.
Reports without content (`""`, `"k.A."`, ...) are skipped.

```json
{
  "train": {
    "ab-1234-5678901234-0": {
      "oct_findings": "Subretinal fluid in the right eye. Drusen in the left eye.",
      "diagnoses": "AMD",
      "age": 74,
      "sex": "Female"
    }
  },
  "val": {
    "ab-1234-9876543210-0": {
      "oct_findings": "No pathological findings."
    }
  }
}
```

## Volumes

For a volume ID `a-b-XXXXXXXX-c` (the third `-`-separated part is used for sharding), the volume is read from

```
<data_root>/<first 2 characters of XXXXXXXX>/<third character of XXXXXXXX>/<volume ID>.h5
```

The HDF5 file contains the dataset `vol`: a `uint8` array with shape `(number of B-scans, height, width)`
(volumes with more than 256 B-scans are uniformly subsampled to 256).

## Weights and language model

* The stage-1 B-scan encoder is read from `_weights/CLIFO.pth` (`run_pretraining_vfe.py`).
* The language model is `Qwen/Qwen3.5-0.8B-Base`, downloaded from the Hugging Face Hub.
* The report-generation metrics (BLEU, ROUGE-L) need the `report` extra: `uv sync --extra report`.
