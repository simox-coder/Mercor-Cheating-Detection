# Mercor Cheating Detection - Pipeline Report

## Overview

This report documents the 4-branch machine learning pipeline for the Mercor Cheating Detection Kaggle competition. The pipeline implements cost-based optimization with threshold search for optimal auto-pass/manual-review/auto-block decisions.

## Metric Parity Proof

All metric tests passed (8/8):
```
============================================================
RUNNING METRIC TESTS
============================================================

✓ Test 1 PASSED: All auto-pass scenario
✓ Test 2 PASSED: All manual review scenario
✓ Test 3 PASSED: Mixed regions scenario
✓ Parity test 1 PASSED: score() matches evaluate()
✓ Parity test 2 PASSED: score_at_thresholds() matches compute_cost()
✓ Parity test 3 PASSED: get_thresholds() matches find_optimal_thresholds()
✓ Edge cases PASSED
✓ Cost constants verified

============================================================
RESULTS: 8/8 tests passed, 0 failed
============================================================

*** ALL TESTS PASSED - METRIC VERIFIED ***
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Train rows | 272,819 |
| Test rows | 48,416 |
| Labeled rows | 112,966 |
| Unlabeled (high_conf_clean) | 159,853 |
| Positive labels (cheaters) | 34,431 |
| Negative labels (clean) | 78,535 |
| Graph nodes | 1,815,244 |
| Graph edges | 1,709,794 |

## Baseline Results (Proxy Public Split)

| Baseline | Cost |
|----------|------|
| constant_0.01 | 2,824,150 |
| constant_0.50 | 2,824,150 |
| constant_0.99 | 2,824,150 |
| uniform_random | 2,824,150 |

## Per-Branch Results

### Branch A: Tabular SOTA Ensemble

**Model:** LightGBM

**Best Parameters:**
```json
{
  "n_estimators": 719,
  "learning_rate": 0.0232,
  "num_leaves": 28,
  "max_depth": 7,
  "min_child_samples": 8,
  "subsample": 0.955,
  "colsample_bytree": 0.629,
  "reg_alpha": 0.00918,
  "reg_lambda": 6.39e-06,
  "scale_pos_weight": 3.08
}
```

**Results:**
| Metric | Value |
|--------|-------|
| CV Cost | 4,400,250 |
| Proxy Public Cost | 2,189,650 |
| Proxy Private Cost | 2,206,700 |

**Features:** 18 original features + 18 isna indicators + 6 row statistics (na_count, mean, std, min, max, range)

---

### Branch B: Graph Embeddings + Tabular

**Model:** LightGBM with graph features

**Best Parameters:**
```json
{
  "n_estimators": 500,
  "learning_rate": 0.03,
  "num_leaves": 63,
  "max_depth": 8,
  "scale_pos_weight": 2.5
}
```

**Results:**
| Metric | Value |
|--------|-------|
| CV Cost | 4,297,750 |
| Proxy Public Cost | 2,145,500 |
| Proxy Private Cost | 2,146,150 |

**Features:** Tabular features + graph features (degree, log_degree, neighbor_degree_mean/median/max/min, component_size, log_component_size)

---

### Branch C: Semi-supervised Graph Propagation (Fold-Safe)

**Model:** LightGBM with fold-safe label propagation features

**Best Parameters:**
```json
{
  "n_estimators": 500,
  "learning_rate": 0.05,
  "num_leaves": 31,
  "max_depth": 6,
  "scale_pos_weight": 2.5
}
```

**Results:**
| Metric | Value |
|--------|-------|
| CV Cost | 4,411,350 |
| Proxy Public Cost | 2,194,250 |
| Proxy Private Cost | 2,210,450 |

**Features:** Tabular features + fold-safe labeled_neighbor_count and labeled_neighbor_cheater_rate (computed using only train fold labels to prevent leakage)

---

### Branch D: Pseudo-GNN (Multi-hop Features)

**Model:** LightGBM with multi-hop graph features

**Best Parameters:**
```json
{
  "n_estimators": 500,
  "learning_rate": 0.03,
  "num_leaves": 63,
  "max_depth": 8,
  "scale_pos_weight": 2.5
}
```

**Results:**
| Metric | Value |
|--------|-------|
| CV Cost | 4,299,350 |
| Proxy Public Cost | 2,143,150 |
| Proxy Private Cost | 2,152,400 |

**Features:** Tabular + graph features + multi-hop features (hop1_count, hop1_log_count, hop1_sum_neighbor_deg, hop1_avg_neighbor_deg, approx_hop2_reach, log_approx_hop2_reach)

---

## Summary Comparison

| Branch | Model | CV Cost | Proxy Public | Proxy Private |
|--------|-------|---------|--------------|---------------|
| A | LightGBM (Tabular) | 4,400,250 | 2,189,650 | 2,206,700 |
| B | LightGBM (Graph) | 4,297,750 | 2,145,500 | 2,146,150 |
| C | LightGBM (Label Prop) | 4,411,350 | 2,194,250 | 2,210,450 |
| **D** | **LightGBM (Multi-hop)** | **4,299,350** | **2,143,150** | **2,152,400** |

**Best Branch by Proxy Public Cost:** Branch D (2,143,150)

## Leakage Prevention

1. **CV Strategy:** StratifiedKFold with 5 folds (SEED=42) on labeled data only
2. **Unlabeled Data:** Used only as pseudo-negatives with small weights (not in validation)
3. **Graph Label Features (Branch C):** Fold-safe computation - only train fold labels used when computing neighbor label statistics
4. **Proxy Split:** Deterministic 50/50 stratified split for public/private proxy evaluation

## Reproduction Commands

### Full Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Run metric tests
python metric_tests.py

# Run full pipeline (all 4 branches)
python run_all_branches.py --data . --out artifacts --budget_fast 15 --budget_bo 10
```

### Verify Outputs
```bash
# Check submissions exist
ls -lh submission_*.csv

# Verify row counts
python -c "import pandas as pd; print(pd.read_csv('test.csv').shape, pd.read_csv('submission_A.csv').shape)"

# View scoreboard
cat artifacts/scoreboard.csv
```

## Output Files

| File | Description |
|------|-------------|
| `submission_A.csv` | Branch A predictions (48,416 rows) |
| `submission_B.csv` | Branch B predictions (48,416 rows) |
| `submission_C.csv` | Branch C predictions (48,416 rows) |
| `submission_D.csv` | Branch D predictions (48,416 rows) |
| `artifacts/scoreboard.csv` | Experiment tracking |
| `artifacts/branch_*/best.json` | Best config per branch |

## Cost Metric Details

- **Miss Cheater Cost:** 500 (auto-passing a cheater)
- **Block Clean Cost:** 100 (auto-blocking a clean user)  
- **Manual Review Cost:** 50 (per sample sent to manual review)

The evaluator searches for optimal thresholds (t_low, t_high) to minimize total cost across three regions:
- Auto-pass: prediction < t_low
- Manual review: t_low ≤ prediction < t_high
- Auto-block: prediction ≥ t_high
