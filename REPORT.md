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

All 7 metric tests pass (4 hand-computable + 3 parity tests):
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

### Basic Pipeline (train.py) - 50 features

**Tabular Features (42 total)**
- `feature_001` to `feature_018`: Original features
- `feature_XXX_missing`: 18 missingness flags
- Row statistics: `na_count`, `row_mean`, `row_std`, `row_min`, `row_max`, `row_range`

**Graph Features (8 total, fold-safe)**
- `degree`: Node degree in full graph
- `log_degree`: Log-transformed degree
- `neighbor_degree_mean/std/min/max`: Neighbor degree statistics
- `n_labeled_neighbors`: Count of labeled neighbors in TRAIN FOLD
- `train_neighbor_cheat_rate`: Cheat rate of TRAIN FOLD neighbors only

### Advanced Pipeline (train_advanced.py) - 69-73 features

**Additional Tabular Features:**
- `row_sum`, `row_median`: Additional row statistics
- `feature_015_log`, `feature_015_bin`: Log transform and binning for wide-range feature
- `feature_010_log`, `feature_010_bin`: Log transform and presence flag
- Binary feature interactions with na_count
- Ratios: `ratio_018_017`, `ratio_high_to_low`

**Additional Graph Features:**
- `degree_sq`: Square root of degree
- `comp_size`, `log_comp_size`: Connected component size
- `is_isolated`: Flag for isolated nodes
- `neighbor_degree_sum`, `neighbor_degree_median`: Additional neighbor stats
- `labeled_ratio`: Proportion of neighbors that are labeled
- Neighbor prediction propagation features (using OOF predictions)

## Models

1. **LightGBM**: GBDT with optimized parameters
2. **XGBoost**: Histogram-based with optimized parameters
3. **CatBoost**: With automatic categorical handling

### Hyperparameters (Advanced Pipeline)
- Learning rate: 0.03
- Max depth/leaves: 7 / 63
- Regularization: L1=0.5, L2=0.5
- Feature/bagging fraction: 0.7
- Early stopping: 100 rounds
- Max iterations: 2000

### Blending
1. **Probability blend**: Simple average of probabilities
2. **Rank blend**: Rank-average of predictions (normalizes to [0,1])

## Results

### Basic Pipeline (train.py)
| Model | CV Cost | CV Score |
|-------|---------|----------|
| LGB | 7,413,515 | -7,413,515 |
| XGB | 7,412,810 | -7,412,810 |
| CAT | 7,419,820 | -7,419,820 |
| BLEND | 7,379,660 | -7,379,660 |

### Advanced Pipeline (train_advanced.py)
| Model | CV Cost | CV Score |
|-------|---------|----------|
| LGB | 7,374,090 | -7,374,090 |
| XGB | 7,405,875 | -7,405,875 |
| CAT | 7,398,075 | -7,398,075 |
| PROB_BLEND | 7,364,750 | -7,364,750 |
| **RANK_BLEND** | **7,363,465** | **-7,363,465** |

### Best Model Results (RANK_BLEND)
- CV Cost: 7,363,465 (for 112,966 samples)
- Per-fold cost: ~1.45-1.5M (for ~22k samples)
- Per-sample cost: 65.18
- Optimal thresholds: t_low=0.5233, t_high=0.9267

### Expected Public LB
- Expected test cost (proportional): ~3,155,901
- Expected public LB score: ~-3,155,901

Note: Public LB ~-1.5M suggests top models achieve about half this cost, indicating significant room for improvement through:
- Better feature engineering
- Semi-supervised learning with pseudo-negatives
- Graph propagation techniques
- Hyperparameter optimization

## Scale Sanity

Baseline sanity checks confirm correct scale:
- Constant predictions → manual-review-all cost (as expected)
- Public LB ~-1.5M corresponds to ~$31/sample for 48k test
- Our CV shows ~$65/sample

## Reproduction Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run metric tests
python metric_tests.py

# Run sanity checks
python sanity_checks.py

# Run basic training
python train.py

# Run advanced training
python train_advanced.py
```

## Output Files

- `submission.csv`: Probability blend submission (48,416 rows)
- `submission_advanced.csv`: Rank blend submission (48,416 rows)
- `artifacts/oof.csv`: Basic pipeline OOF predictions
- `artifacts/oof_advanced.csv`: Advanced pipeline OOF predictions
- `artifacts/cv_summary.json`: Basic CV summary
- `artifacts/cv_summary_advanced.json`: Advanced CV summary

## Future Improvements

1. **Semi-supervised learning**: Use `high_conf_clean` rows as pseudo-negatives
2. **Graph propagation**: Propagate predictions through social graph
3. **Node embeddings**: Node2Vec or DeepWalk on social graph
4. **Hyperparameter tuning**: Optuna optimization for cost metric
5. **Calibration**: Isotonic/Platt calibration if it improves cost
