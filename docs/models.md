# Pretrained models and predictions

Causal Colon-Audit-Mamba (12 blocks, EndoMamba initialization, whole-procedure memory), checkpoint selected on
validation MCC. Five training runs (seeds 0–4) per fold, for the official REAL-Colon 5-fold and cross-center
4-fold protocols. These are the models behind the paper's results for the final model.

## Checkpoints

One zip per fold with the five runs (seeds 0–4), about 225 MB each. Each `best-model-val_mcc.pt` is a PyTorch
`state_dict` (about 45 MB). The model code that loads them will be released when the paper is accepted.

| Protocol | Fold | Download |
|---|---|---|
| 5-fold | 1 | [checkpoints_5fold_fold1.zip](https://drive.google.com/uc?id=14Mn5XwgQxMfmehLDdJwfjzTII6_W9S99) |
| 5-fold | 2 | [checkpoints_5fold_fold2.zip](https://drive.google.com/uc?id=1KrK1g1SE3PZVTeuIWohBRVQPQmAP1_FI) |
| 5-fold | 3 | [checkpoints_5fold_fold3.zip](https://drive.google.com/uc?id=1F_WAonbsNzjLRbJdpbk0W90j20rUrGGP) |
| 5-fold | 4 | [checkpoints_5fold_fold4.zip](https://drive.google.com/uc?id=1DrriOnyEG9r5--i0ATx2dVVJD_Y-0kTL) |
| 5-fold | 5 | [checkpoints_5fold_fold5.zip](https://drive.google.com/uc?id=1dhpDLehpw_96ezLbMD9FzBrqsBlZhbWK) |
| 4-fold | 1 | [checkpoints_4fold_fold1.zip](https://drive.google.com/uc?id=1LlveR7KamBIhVLhq_2gHMZLrdjSh-B3U) |
| 4-fold | 2 | [checkpoints_4fold_fold2.zip](https://drive.google.com/uc?id=11nj5KhiGf3Yn6Yrgj0Kj3amO8vYBGWks) |
| 4-fold | 3 | [checkpoints_4fold_fold3.zip](https://drive.google.com/uc?id=1L4eglLmrH5Dp2T3Ef2ivduKL5m8do1X-) |
| 4-fold | 4 | [checkpoints_4fold_fold4.zip](https://drive.google.com/uc?id=1XjZp4AofgAozPfWFKSoBe7-TltHXzI1R) |

Unzip into `OUTPUT_DIR`. The expected layout is:

```
OUTPUT_DIR/mamba/<protocol>/fold<k>/mamba12-endomamba/seed<s>/best-model-val_mcc.pt
```

## Frame-level predictions

Validation and test predictions of every Mamba run behind Table 1 and Table 2 (the final model, the memory-limited,
scratch and bidirectional variants, and the other depths and augmentations).
`evaluation/paper_tables.py` and `analysis/figure_timeline.py` read them directly.

| Protocol | Download |
|---|---|
| 5-fold | [predictions_5fold.zip](https://drive.google.com/uc?id=1cY5gL8mch2ZT8YwOAW_3h3w1IJL60-HQ) |
| 4-fold | [predictions_4fold.zip](https://drive.google.com/uc?id=152Noto0BHq-AIZmJmo3YAzYk0HHgUiG1) |

Unzip both into `OUTPUT_DIR`. The expected layout is:

```
OUTPUT_DIR/mamba/<protocol>/fold<k>/<variant>/seed<s>/frame_predictions_{valid,test}_val_mcc.json
```

**Columns:**
- **`patient, frame`:** REAL-Colon video and native frame index.
- **`anatomy`:** ground-truth class.
- **`PredictedClass`:** online (causal) prediction, or offline for the bidirectional variant.
- **`near_lesion`:** 1 if the frame is within ±30 s of an annotated lesion.

The class codes are listed in the main [README](../README.md#pretrained-models-and-predictions).
