# Mercor Cheating Detection - Solution Report

## Overview

This solution implements a leakage-safe machine learning pipeline for the Mercor Cheating Detection Kaggle competition. The goal is to predict whether a user is cheating based on tabular features and social graph connections, optimizing for a cost-based evaluation metric.

## Metric

The competition uses a cost-based metric with 3 decision regions:
- **Auto-pass** (prediction < t_low): Low-risk users pass automatically
- **Manual review** (t_low ≤ prediction < t_high): Medium-risk users sent for review
- **Auto-block** (prediction ≥ t_high): High-risk users blocked automatically

**Costs:**
- Cheater auto-passed (FN): 600
- Non-cheater auto-blocked (FP): 300
- Non-cheater in manual review: 150
- Cheater in manual review: 5
- Correct auto-pass/block: 0

**Kaggle Score = NEGATIVE of minimum total cost**

### Metric Parity Proof

The official evaluation logic is implemented in `official_eval_reference.py` with SHA256 hash:
```
SHA256 of core evaluation logic (lines 45-95): 06172356a1221d68a5b64ec0c26ed6cf5fbe096ef79edef3f3848bcf6ae7341a
```

All 7 metric tests pass (3 hand-computable + 4 parity tests):
```
python metric_tests.py
SUMMARY: 7 passed, 0 failed
```

## Data

- **Train rows:** 272,819 (112,966 labeled, 159,853 unlabeled)
- **Test rows:** 48,416
- **Labeled cheaters:** 34,431 (30.5%)
- **Labeled non-cheaters:** 78,535
- **Graph edges:** 1,714,763

## Cross-Validation Scheme

**Why it's leakage-safe:**
1. StratifiedKFold (5 folds, SEED=42) on LABELED rows only
2. Graph features use ONLY train fold labels (no validation labels)
3. Neighbor cheat rates computed from train fold only
4. Unlabeled rows never in validation

## Features

### Tabular Features (42 total)
- `feature_001` to `feature_018`: Original features
- `feature_XXX_missing`: 18 missingness flags
- Row statistics: `na_count`, `row_mean`, `row_std`, `row_min`, `row_max`, `row_range`

### Graph Features (8 total, fold-safe)
- `degree`: Node degree in full graph
- `log_degree`: Log-transformed degree
- `neighbor_degree_mean/std/min/max`: Neighbor degree statistics
- `n_labeled_neighbors`: Count of labeled neighbors in TRAIN FOLD
- `train_neighbor_cheat_rate`: Cheat rate of TRAIN FOLD neighbors only

## Models

1. **LightGBM**: GBDT with early stopping
2. **XGBoost**: Histogram-based with early stopping  
3. **CatBoost**: With automatic handling of categorical features

### Blending
Simple probability average: `(pred_lgb + pred_xgb + pred_cat) / 3`

## Results

### CV Results (Full OOF)
| Model | Cost | Score |
|-------|------|-------|
| LGB | 7,413,515 | -7,413,515 |
| XGB | 7,412,810 | -7,412,810 |
| CAT | 7,419,820 | -7,419,820 |
| BLEND | 7,379,660 | -7,379,660 |

Note: CV cost is for 112,966 samples. Per-fold cost is ~1.45-1.5M for ~22k samples.

### Optimal Thresholds (Blend)
- t_low: 0.2000
- t_high: 0.9550

### Decision Regions (Blend)
- Auto-pass: 58,213
- Manual review: 46,993
- Auto-block: 7,760

### Cost Breakdown (Blend)
- FN auto-pass: 5,627 × 600 = 3,376,200
- FP auto-block: 38 × 300 = 11,400
- FP manual: 25,911 × 150 = 3,886,650
- TP manual: 21,082 × 5 = 105,410

## Reproduction Commands

```bash
# Install dependencies
pip install pandas numpy scikit-learn lightgbm xgboost catboost

# Run metric tests
python metric_tests.py

# Run sanity checks
python sanity_checks.py

# Train and generate submission
python train.py
```

## Output Files

- `submission.csv`: Final submission (48,416 rows)
- `artifacts/oof.csv`: Out-of-fold predictions
- `artifacts/cv_summary.json`: Cross-validation summary

## Scale Sanity

Baseline sanity checks confirm correct scale:
- Constant predictions → manual-review-all cost (as expected)
- Public LB ~-1.5M corresponds to ~$31/sample for 48k test
- Our CV shows ~$65/sample, indicating room for improvement

## Notes on Semi-Supervised Learning

The `high_conf_clean` unlabeled rows (159,853) can potentially be used as pseudo-negatives. Current implementation uses labeled rows only. Future improvements could explore:
- Pseudo-labeling with confidence thresholds
- PU learning techniques
- Self-training with careful weight tuning
