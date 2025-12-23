"""Configuration and constants for the pipeline."""
import os

# Paths
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")

# Data files
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
GRAPH_FILE = os.path.join(DATA_DIR, "social_graph.csv")
SAMPLE_SUB_FILE = os.path.join(DATA_DIR, "sample_submission.csv")
METADATA_FILE = os.path.join(DATA_DIR, "feature_metadata.json")

# Feature columns
FEATURE_COLS = [f"feature_{i:03d}" for i in range(1, 19)]
ID_COL = "user_hash"
LABEL_COL = "is_cheating"
HIGH_CONF_CLEAN_COL = "high_conf_clean"

# Random seed for reproducibility
SEED = 42

# Cost metric parameters (from competition description)
# Cost regions:
# auto-pass (pred < low_thresh): cost = 500 if cheater, 0 if clean
# manual review (low_thresh <= pred < high_thresh): cost = 25 for all
# auto-block (pred >= high_thresh): cost = 0 if cheater, 100 if clean
COST_FN = 500  # False negative cost (auto-pass a cheater)
COST_REVIEW = 25  # Manual review cost
COST_FP = 100  # False positive cost (auto-block a clean user)

# Node2Vec parameters
NODE2VEC_DIM = 64
NODE2VEC_WALK_LENGTH = 30
NODE2VEC_NUM_WALKS = 20
NODE2VEC_P = 1.0
NODE2VEC_Q = 1.0
NODE2VEC_WORKERS = 4

# CV settings
N_FOLDS = 5

# Model hyperparameters (defaults, will be tuned)
CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": SEED,
    "verbose": 100,
    "early_stopping_rounds": 100,
    "task_type": "CPU",
    "loss_function": "Logloss",
}

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_child_samples": 20,
    "seed": SEED,
    "verbose": -1,
    "n_estimators": 2000,
}

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "seed": SEED,
    "verbosity": 0,
    "n_estimators": 2000,
}

# PU learning weight for high_conf_clean pseudo-negatives
PU_WEIGHT = 0.05

# TARGET LB score (placeholder, will be updated)
TARGET_PUBLIC_SCORE = -1540000
