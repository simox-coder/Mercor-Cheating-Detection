# Mercor Cheating Detection - Competition Report

## Overview
This repository contains a comprehensive ML pipeline for the Mercor Cheating Detection Kaggle competition. The goal is to identify cheating users based on tabular features and a social graph, optimizing for a cost-based metric with three regions (auto-pass, manual review, auto-block).

## Pipeline Architecture

### 1. Data Loading & Validation (`src/data_loader.py`)
- Loads train.csv (272,819 rows), test.csv (48,416 rows), social_graph.csv (1.7M edges)
- Validates column presence and data types
- Separates labeled (112,966) and unlabeled (159,853 high_conf_clean) data

### 2. Graph Feature Engineering (`src/graph_features.py`)
- **Degree features**: node degree, log-degree
- **Component features**: connected component ID and size using Union-Find
- **Centrality**: PageRank approximation for large graphs
- **Clustering coefficient**: local clustering approximation
- **Node2Vec embeddings**: 32-64 dimensional embeddings via random walks + Word2Vec

### 3. Tabular Feature Engineering
- **Missing indicators**: Binary flags for each missing feature
- **Feature interactions**: Selected ratios and sums
- **Total missing count**: Aggregate missingness signal

### 4. Cross-Validation (`src/cv_utils.py`)
- **Component-based GroupKFold**: Splits by connected components to prevent graph leakage
- Users in the same component are always in the same fold
- This prevents neighbor-based features from leaking validation labels

### 5. Models (`src/models.py`)
- **CatBoost**: 3000 iterations, depth=8, lr=0.03
- **LightGBM**: 3000 estimators, 63 leaves, lr=0.03
- **XGBoost**: 3000 estimators, depth=8, eta=0.03
- All with early stopping (200 rounds)

### 6. PU Learning
- High_conf_clean samples treated as pseudo-negatives with weight=0.05
- Class balancing applied to labeled data
- Conservative approach to avoid assuming all unlabeled are clean

### 7. Ensemble & Blending (`src/blending.py`)
- Grid search over blend weights
- Three blending methods: linear, rank-based, logit-space
- Selection based on local cost metric

### 8. Score Propagation (`src/propagation.py`)
- Simple diffusion: `new_score = (1-α)*score + α*mean(neighbor_scores)`
- PageRank-style diffusion with personalization
- Multiple alpha values tested (0.1-0.3)

### 9. Local Evaluator (`src/evaluator.py`)
- Implements official cost metric:
  - Auto-pass (pred < low_thresh): 500 cost per false negative
  - Manual review (low_thresh ≤ pred < high_thresh): 25 cost per user
  - Auto-block (pred ≥ high_thresh): 100 cost per false positive
- Searches optimal threshold pair to minimize total cost

## Leakage Prevention
1. **Component-based CV**: No graph neighbors across train/val split
2. **OOF predictions only**: Neighbor aggregations use out-of-fold predictions
3. **No test label inference**: Strictly supervised on train labels only

## Commands to Reproduce

### Full Training
```bash
python run_pipeline.py
```

### Standard Training
```bash
python train.py --data . --out artifacts/run_001
```

### Inference
```bash
python infer.py --data . --ckpt artifacts/run_001 --out submission.csv
```

### Evaluate OOF
```bash
python tools/eval.py --oof artifacts/run_001/oof/oof_predictions.csv
```

## Files Structure
```
├── AGENTS.md                 # Competition requirements
├── REPORT.md                 # This file
├── train.py                  # Training entrypoint
├── infer.py                  # Inference entrypoint
├── run_pipeline.py           # Full pipeline script
├── src/
│   ├── config.py            # Configuration
│   ├── data_loader.py       # Data loading
│   ├── graph_features.py    # Graph feature engineering
│   ├── evaluator.py         # Cost metric evaluation
│   ├── cv_utils.py          # Cross-validation utilities
│   ├── models.py            # Model training
│   ├── propagation.py       # Score propagation
│   └── blending.py          # Ensemble blending
├── tools/
│   └── eval.py              # OOF evaluation tool
├── artifacts/
│   ├── leaderboard_target.json
│   └── scoreboard.csv
└── data files (train.csv, test.csv, social_graph.csv, etc.)
```

## Key Hyperparameters
- PU weight: 0.05 for high_conf_clean pseudo-negatives
- CV folds: 5 (component-based)
- Node2Vec: dim=32, walks=5, length=15
- Models: 3000 iterations with early stopping at 200

## Notes
- Graph has 1.7M nodes including "ghost nodes" not in train/test
- 17,624 connected components found
- Cheating rate in labeled data: ~30.5%
- Target LB score placeholder: -1,540,000 (update with actual #1 score)
