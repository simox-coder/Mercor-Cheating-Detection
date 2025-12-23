#!/usr/bin/env python3
"""
Mercor Cheating Detection - Full Training Pipeline
===================================================
Optimizes for COST-BASED metric, NOT AUC.

Cost structure:
- False Negative (cheater auto-pass): $600
- False Positive in auto-block: $300
- False Positive in manual review: $150
- True Positive in manual review: $5
- Correct auto-pass / auto-block: $0

Final score = -min_total_cost (higher is better on leaderboard)
"""

import os
import json
import warnings
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
import networkx as nx

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42
N_FOLDS = 5
DATA_DIR = Path('/home/runner/work/Mercor-Cheating-Detection/Mercor-Cheating-Detection')
ARTIFACTS_DIR = DATA_DIR / 'artifacts'
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Cost structure from official evaluation
COST_FN = 600  # False Negative (cheater auto-pass)
COST_FP_BLOCK = 300  # False Positive in auto-block
COST_FP_REVIEW = 150  # False Positive in manual review
COST_TP_REVIEW = 5    # True Positive in manual review

# Feature columns
FEATURE_COLS = [f'feature_{str(i).zfill(3)}' for i in range(1, 19)]

np.random.seed(SEED)

# ============================================================================
# OFFICIAL COST METRIC IMPLEMENTATION
# ============================================================================
def compute_cost(y_true: np.ndarray, y_pred: np.ndarray, 
                 t_low: float, t_high: float) -> Dict:
    """
    Compute total cost for given thresholds.
    
    Decision regions:
    - Auto-pass (low risk): p < t_low
    - Manual review (medium risk): t_low <= p <= t_high  
    - Auto-block (high risk): p > t_high
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Decision regions
    auto_pass = y_pred < t_low
    manual_review = (y_pred >= t_low) & (y_pred <= t_high)
    auto_block = y_pred > t_high
    
    # Count outcomes
    # Auto-pass region
    fn_count = np.sum((y_true == 1) & auto_pass)  # Cheater auto-passed
    tn_count = np.sum((y_true == 0) & auto_pass)  # Non-cheater auto-passed
    
    # Auto-block region
    tp_block_count = np.sum((y_true == 1) & auto_block)  # Cheater blocked
    fp_block_count = np.sum((y_true == 0) & auto_block)  # Non-cheater blocked
    
    # Manual review region
    tp_review_count = np.sum((y_true == 1) & manual_review)
    fp_review_count = np.sum((y_true == 0) & manual_review)
    
    # Compute costs
    fn_cost = fn_count * COST_FN
    fp_block_cost = fp_block_count * COST_FP_BLOCK
    fp_review_cost = fp_review_count * COST_FP_REVIEW
    tp_review_cost = tp_review_count * COST_TP_REVIEW
    
    total_cost = fn_cost + fp_block_cost + fp_review_cost + tp_review_cost
    
    return {
        'total_cost': total_cost,
        'fn_cost': fn_cost,
        'fp_block_cost': fp_block_cost,
        'fp_review_cost': fp_review_cost,
        'tp_review_cost': tp_review_cost,
        'fn_count': fn_count,
        'fp_block_count': fp_block_count,
        'fp_review_count': fp_review_count,
        'tp_review_count': tp_review_count,
        'tn_count': tn_count,
        'tp_block_count': tp_block_count,
        'auto_pass_count': int(auto_pass.sum()),
        'manual_review_count': int(manual_review.sum()),
        'auto_block_count': int(auto_block.sum()),
        't_low': t_low,
        't_high': t_high
    }


def find_optimal_thresholds(y_true: np.ndarray, y_pred: np.ndarray, 
                            n_steps: int = 100) -> Tuple[float, float, Dict]:
    """
    Grid search for optimal threshold pair (t_low, t_high) that minimizes cost.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create threshold candidates
    thresholds = np.linspace(0, 1, n_steps + 1)
    
    best_cost = float('inf')
    best_t_low = 0.0
    best_t_high = 1.0
    best_result = None
    
    for t_low in thresholds:
        for t_high in thresholds:
            if t_high < t_low:
                continue
            
            result = compute_cost(y_true, y_pred, t_low, t_high)
            if result['total_cost'] < best_cost:
                best_cost = result['total_cost']
                best_t_low = t_low
                best_t_high = t_high
                best_result = result
    
    return best_t_low, best_t_high, best_result


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                         verbose: bool = True) -> Tuple[float, Dict]:
    """
    Evaluate predictions using the official cost metric.
    Returns negative cost (higher is better, matching leaderboard).
    """
    t_low, t_high, result = find_optimal_thresholds(y_true, y_pred)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"COST METRIC EVALUATION")
        print(f"{'='*60}")
        print(f"Optimal thresholds: t_low={t_low:.4f}, t_high={t_high:.4f}")
        print(f"\nDecision counts:")
        print(f"  Auto-pass:      {result['auto_pass_count']:,}")
        print(f"  Manual review:  {result['manual_review_count']:,}")
        print(f"  Auto-block:     {result['auto_block_count']:,}")
        print(f"\nCost breakdown:")
        print(f"  FN cost (cheater auto-pass):     ${result['fn_cost']:,.0f} ({result['fn_count']} cases)")
        print(f"  FP_block cost (wrong block):     ${result['fp_block_cost']:,.0f} ({result['fp_block_count']} cases)")
        print(f"  FP_review cost (wrong review):   ${result['fp_review_cost']:,.0f} ({result['fp_review_count']} cases)")
        print(f"  TP_review cost (correct review): ${result['tp_review_cost']:,.0f} ({result['tp_review_count']} cases)")
        print(f"\n  TOTAL COST: ${result['total_cost']:,.0f}")
        print(f"  LEADERBOARD SCORE: {-result['total_cost']:,.0f}")
        print(f"{'='*60}\n")
    
    return -result['total_cost'], result


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and social graph data."""
    print("Loading data...")
    
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    social_graph = pd.read_csv(DATA_DIR / 'social_graph.csv')
    
    print(f"  Train: {train.shape}")
    print(f"  Test: {test.shape}")
    print(f"  Social graph edges: {len(social_graph):,}")
    
    return train, test, social_graph


def create_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tabular features including missingness indicators.
    """
    df = df.copy()
    
    # Create missingness indicators for each feature
    for col in FEATURE_COLS:
        df[f'{col}_isna'] = df[col].isna().astype(np.float32)
    
    # Total missing count per row
    df['na_count'] = df[FEATURE_COLS].isna().sum(axis=1)
    
    # Row statistics (for non-missing values)
    numeric_vals = df[FEATURE_COLS].values
    
    # Row mean (ignoring NaN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df['row_mean'] = np.nanmean(numeric_vals, axis=1)
        df['row_std'] = np.nanstd(numeric_vals, axis=1)
        df['row_min'] = np.nanmin(numeric_vals, axis=1)
        df['row_max'] = np.nanmax(numeric_vals, axis=1)
        df['row_range'] = df['row_max'] - df['row_min']
    
    # Fill NaN with median for training (but isna flags preserve information)
    for col in FEATURE_COLS:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    # Fill any remaining NaN in derived features
    df['row_mean'] = df['row_mean'].fillna(0)
    df['row_std'] = df['row_std'].fillna(0)
    df['row_min'] = df['row_min'].fillna(0)
    df['row_max'] = df['row_max'].fillna(0)
    df['row_range'] = df['row_range'].fillna(0)
    
    return df


def create_graph_features(train: pd.DataFrame, test: pd.DataFrame, 
                          social_graph: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create graph-based features from social_graph.
    """
    print("Creating graph features...")
    
    # Combine all users
    all_train_users = set(train['user_hash'].unique())
    all_test_users = set(test['user_hash'].unique())
    all_users = all_train_users | all_test_users
    
    # Build graph
    G = nx.Graph()
    G.add_nodes_from(all_users)
    
    edges = social_graph[['user_a', 'user_b']].values.tolist()
    G.add_edges_from(edges)
    
    print(f"  Graph nodes: {G.number_of_nodes():,}")
    print(f"  Graph edges: {G.number_of_edges():,}")
    
    # Create labeled user lookup for neighbor aggregation
    labeled_mask = train['is_cheating'].notna()
    labeled_users = train.loc[labeled_mask, ['user_hash', 'is_cheating']].copy()
    cheating_lookup = dict(zip(labeled_users['user_hash'], labeled_users['is_cheating']))
    
    def compute_node_features(user_hash):
        """Compute graph features for a single user."""
        features = {}
        
        # Degree
        if user_hash in G:
            neighbors = list(G.neighbors(user_hash))
            features['degree'] = len(neighbors)
            features['log_degree'] = np.log1p(len(neighbors))
            
            # Neighbor statistics
            train_neighbors = [n for n in neighbors if n in all_train_users]
            test_neighbors = [n for n in neighbors if n in all_test_users]
            labeled_neighbors = [n for n in neighbors if n in cheating_lookup]
            
            features['n_train_neighbors'] = len(train_neighbors)
            features['n_test_neighbors'] = len(test_neighbors)
            features['n_labeled_neighbors'] = len(labeled_neighbors)
            
            # Fraction of labeled neighbors
            if len(neighbors) > 0:
                features['frac_labeled_neighbors'] = len(labeled_neighbors) / len(neighbors)
            else:
                features['frac_labeled_neighbors'] = 0.0
            
            # Mean cheating rate of labeled neighbors (CAREFUL: this must be fold-safe in CV)
            if len(labeled_neighbors) > 0:
                cheating_rates = [cheating_lookup[n] for n in labeled_neighbors]
                features['neighbor_mean_cheating'] = np.mean(cheating_rates)
                features['neighbor_sum_cheating'] = np.sum(cheating_rates)
                features['neighbor_max_cheating'] = np.max(cheating_rates)
            else:
                features['neighbor_mean_cheating'] = 0.0
                features['neighbor_sum_cheating'] = 0.0
                features['neighbor_max_cheating'] = 0.0
        else:
            features = {
                'degree': 0,
                'log_degree': 0,
                'n_train_neighbors': 0,
                'n_test_neighbors': 0,
                'n_labeled_neighbors': 0,
                'frac_labeled_neighbors': 0.0,
                'neighbor_mean_cheating': 0.0,
                'neighbor_sum_cheating': 0.0,
                'neighbor_max_cheating': 0.0
            }
        
        return features
    
    # Compute features for train
    print("  Computing graph features for train...")
    train_graph_features = []
    for user_hash in train['user_hash']:
        train_graph_features.append(compute_node_features(user_hash))
    train_graph_df = pd.DataFrame(train_graph_features)
    
    # Compute features for test
    print("  Computing graph features for test...")
    test_graph_features = []
    for user_hash in test['user_hash']:
        test_graph_features.append(compute_node_features(user_hash))
    test_graph_df = pd.DataFrame(test_graph_features)
    
    # Merge with original dataframes
    for col in train_graph_df.columns:
        train[col] = train_graph_df[col].values
        test[col] = test_graph_df[col].values
    
    print(f"  Added {len(train_graph_df.columns)} graph features")
    
    return train, test


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, 
                     social_graph: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Full feature preparation pipeline.
    """
    # Create tabular features
    print("\nCreating tabular features...")
    train = create_tabular_features(train)
    test = create_tabular_features(test)
    
    # Create graph features
    train, test = create_graph_features(train, test, social_graph)
    
    # Define feature columns (excluding user_hash, is_cheating, high_conf_clean)
    exclude_cols = ['user_hash', 'is_cheating', 'high_conf_clean']
    feature_cols = [c for c in train.columns if c not in exclude_cols]
    
    print(f"\nTotal features: {len(feature_cols)}")
    
    return train, test, feature_cols


# ============================================================================
# MODEL TRAINING
# ============================================================================
def get_lightgbm_params(trial: Optional[optuna.Trial] = None) -> Dict:
    """Get LightGBM parameters, optionally tuned by Optuna."""
    if trial is not None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'verbose': -1,
            'seed': SEED,
            'n_jobs': -1
        }
    else:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 64,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'max_depth': 8,
            'verbose': -1,
            'seed': SEED,
            'n_jobs': -1
        }
    return params


def get_xgboost_params(trial: Optional[optuna.Trial] = None) -> Dict:
    """Get XGBoost parameters, optionally tuned by Optuna."""
    if trial is not None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 100),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': 0
        }
    else:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': 0
        }
    return params


def get_catboost_params(trial: Optional[optuna.Trial] = None) -> Dict:
    """Get CatBoost parameters, optionally tuned by Optuna."""
    if trial is not None:
        params = {
            'loss_function': 'Logloss',
            'iterations': 1000,
            'depth': trial.suggest_int('cb_depth', 4, 10),
            'learning_rate': trial.suggest_float('cb_learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1.0, 10.0),
            'border_count': trial.suggest_int('cb_border_count', 32, 255),
            'random_strength': trial.suggest_float('cb_random_strength', 0.0, 10.0),
            'bagging_temperature': trial.suggest_float('cb_bagging_temperature', 0.0, 1.0),
            'random_seed': SEED,
            'verbose': False,
            'allow_writing_files': False
        }
    else:
        params = {
            'loss_function': 'Logloss',
            'iterations': 1000,
            'depth': 6,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3.0,
            'border_count': 128,
            'random_strength': 1.0,
            'bagging_temperature': 0.5,
            'random_seed': SEED,
            'verbose': False,
            'allow_writing_files': False
        }
    return params


def train_fold_lightgbm(X_train, y_train, X_val, y_val, sample_weight=None, params=None):
    """Train a single fold with LightGBM."""
    if params is None:
        params = get_lightgbm_params()
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    return model


def train_fold_xgboost(X_train, y_train, X_val, y_val, sample_weight=None, params=None):
    """Train a single fold with XGBoost."""
    if params is None:
        params = get_xgboost_params()
    
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    return model


def train_fold_catboost(X_train, y_train, X_val, y_val, sample_weight=None, params=None):
    """Train a single fold with CatBoost."""
    if params is None:
        params = get_catboost_params()
    
    model = cb.CatBoostClassifier(**params)
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )
    
    return model


# ============================================================================
# CROSS-VALIDATION WITH COST OPTIMIZATION
# ============================================================================
def cross_validate_with_cost(train: pd.DataFrame, feature_cols: List[str],
                             pseudo_weight: float = 0.05,
                             lgb_params=None, xgb_params=None, cb_params=None,
                             blend_weights=None,
                             verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Stratified K-Fold CV optimizing for cost metric.
    
    Uses labeled data for CV, with optional pseudo-negatives from high_conf_clean.
    """
    # Separate labeled and unlabeled data
    labeled_mask = train['is_cheating'].notna()
    labeled_data = train[labeled_mask].copy()
    unlabeled_data = train[~labeled_mask].copy()  # high_conf_clean=1 unlabeled
    
    X_labeled = labeled_data[feature_cols].values
    y_labeled = labeled_data['is_cheating'].values
    
    if verbose:
        print(f"\nCross-validation setup:")
        print(f"  Labeled samples: {len(labeled_data):,}")
        print(f"  Unlabeled (pseudo-neg): {len(unlabeled_data):,}")
        print(f"  Pseudo-negative weight: {pseudo_weight}")
        print(f"  Positive class ratio: {y_labeled.mean():.4f}")
    
    # Prepare pseudo-negatives
    X_pseudo = unlabeled_data[feature_cols].values
    y_pseudo = np.zeros(len(unlabeled_data))  # Treat as negatives
    
    # Initialize OOF predictions
    oof_preds_lgb = np.zeros(len(labeled_data))
    oof_preds_xgb = np.zeros(len(labeled_data))
    oof_preds_cb = np.zeros(len(labeled_data))
    
    # Store models for later
    models_lgb = []
    models_xgb = []
    models_cb = []
    
    fold_costs = []
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
        if verbose:
            print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        
        X_train_fold = X_labeled[train_idx]
        y_train_fold = y_labeled[train_idx]
        X_val_fold = X_labeled[val_idx]
        y_val_fold = y_labeled[val_idx]
        
        # Add pseudo-negatives with weight
        X_train_aug = np.vstack([X_train_fold, X_pseudo])
        y_train_aug = np.hstack([y_train_fold, y_pseudo])
        
        # Sample weights: 1.0 for labeled, pseudo_weight for unlabeled
        weights = np.hstack([
            np.ones(len(train_idx)),
            np.full(len(y_pseudo), pseudo_weight)
        ])
        
        # Train LightGBM
        model_lgb = train_fold_lightgbm(
            X_train_aug, y_train_aug, X_val_fold, y_val_fold,
            sample_weight=weights, params=lgb_params
        )
        models_lgb.append(model_lgb)
        oof_preds_lgb[val_idx] = model_lgb.predict(X_val_fold)
        
        # Train XGBoost
        model_xgb = train_fold_xgboost(
            X_train_aug, y_train_aug, X_val_fold, y_val_fold,
            sample_weight=weights, params=xgb_params
        )
        models_xgb.append(model_xgb)
        dval = xgb.DMatrix(X_val_fold)
        oof_preds_xgb[val_idx] = model_xgb.predict(dval)
        
        # Train CatBoost
        model_cb = train_fold_catboost(
            X_train_aug, y_train_aug, X_val_fold, y_val_fold,
            sample_weight=weights, params=cb_params
        )
        models_cb.append(model_cb)
        oof_preds_cb[val_idx] = model_cb.predict_proba(X_val_fold)[:, 1]
        
        # Evaluate fold cost (for blended predictions)
        if blend_weights is None:
            blend_weights = [0.4, 0.3, 0.3]  # LGB, XGB, CB
        
        fold_pred = (
            blend_weights[0] * oof_preds_lgb[val_idx] +
            blend_weights[1] * oof_preds_xgb[val_idx] +
            blend_weights[2] * oof_preds_cb[val_idx]
        )
        
        _, fold_result = evaluate_predictions(y_val_fold, fold_pred, verbose=False)
        fold_costs.append(fold_result['total_cost'])
        
        if verbose:
            print(f"  Fold cost: ${fold_result['total_cost']:,.0f}")
    
    # Compute final blended OOF
    if blend_weights is None:
        blend_weights = [0.4, 0.3, 0.3]
    
    oof_blended = (
        blend_weights[0] * oof_preds_lgb +
        blend_weights[1] * oof_preds_xgb +
        blend_weights[2] * oof_preds_cb
    )
    
    # Final CV score
    cv_score, cv_result = evaluate_predictions(y_labeled, oof_blended, verbose=verbose)
    
    results = {
        'cv_cost': cv_result['total_cost'],
        'cv_score': cv_score,
        'fold_costs': fold_costs,
        'mean_fold_cost': np.mean(fold_costs),
        'std_fold_cost': np.std(fold_costs),
        't_low': cv_result['t_low'],
        't_high': cv_result['t_high'],
        'models_lgb': models_lgb,
        'models_xgb': models_xgb,
        'models_cb': models_cb,
        'oof_preds_lgb': oof_preds_lgb,
        'oof_preds_xgb': oof_preds_xgb,
        'oof_preds_cb': oof_preds_cb,
        'oof_blended': oof_blended,
        'y_labeled': y_labeled,
        'labeled_user_hashes': labeled_data['user_hash'].values,
        'blend_weights': blend_weights,
        'pseudo_weight': pseudo_weight
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CV SUMMARY")
        print(f"{'='*60}")
        print(f"Mean fold cost: ${np.mean(fold_costs):,.0f} ± ${np.std(fold_costs):,.0f}")
        print(f"Overall CV cost: ${cv_result['total_cost']:,.0f}")
        print(f"CV Score (LB format): {cv_score:,.0f}")
        print(f"{'='*60}\n")
    
    return oof_blended, results


# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================
def optuna_objective(trial: optuna.Trial, train: pd.DataFrame, 
                     feature_cols: List[str]) -> float:
    """
    Optuna objective function - minimize CV cost.
    """
    # Hyperparameters to tune
    pseudo_weight = trial.suggest_float('pseudo_weight', 0.01, 0.15)
    
    lgb_params = get_lightgbm_params(trial)
    xgb_params = get_xgboost_params(trial)
    cb_params = get_catboost_params(trial)
    
    # Blend weights
    w_lgb = trial.suggest_float('w_lgb', 0.2, 0.6)
    w_xgb = trial.suggest_float('w_xgb', 0.1, 0.5)
    w_cb = 1.0 - w_lgb - w_xgb
    
    if w_cb < 0.1:
        return float('inf')
    
    blend_weights = [w_lgb, w_xgb, w_cb]
    
    # Run CV
    _, results = cross_validate_with_cost(
        train, feature_cols,
        pseudo_weight=pseudo_weight,
        lgb_params=lgb_params,
        xgb_params=xgb_params,
        cb_params=cb_params,
        blend_weights=blend_weights,
        verbose=False
    )
    
    return results['cv_cost']


def optimize_hyperparameters(train: pd.DataFrame, feature_cols: List[str],
                             n_trials: int = 50) -> Dict:
    """
    Run Optuna hyperparameter optimization.
    """
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER OPTIMIZATION ({n_trials} trials)")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=SEED)
    )
    
    study.optimize(
        lambda trial: optuna_objective(trial, train, feature_cols),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\nBest trial:")
    print(f"  Cost: ${study.best_value:,.0f}")
    print(f"  Params: {study.best_params}")
    
    # Save study
    study_path = ARTIFACTS_DIR / 'optuna_study.pkl'
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    
    return study.best_params


# ============================================================================
# INFERENCE
# ============================================================================
def predict_test(test: pd.DataFrame, feature_cols: List[str],
                 results: Dict) -> np.ndarray:
    """
    Generate predictions for test set using trained models.
    """
    print("\nGenerating test predictions...")
    
    X_test = test[feature_cols].values
    
    # Average predictions from all folds
    test_preds_lgb = np.zeros(len(test))
    test_preds_xgb = np.zeros(len(test))
    test_preds_cb = np.zeros(len(test))
    
    for fold, (model_lgb, model_xgb, model_cb) in enumerate(
        zip(results['models_lgb'], results['models_xgb'], results['models_cb'])):
        
        test_preds_lgb += model_lgb.predict(X_test) / N_FOLDS
        
        dtest = xgb.DMatrix(X_test)
        test_preds_xgb += model_xgb.predict(dtest) / N_FOLDS
        
        test_preds_cb += model_cb.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # Blend
    blend_weights = results['blend_weights']
    test_blended = (
        blend_weights[0] * test_preds_lgb +
        blend_weights[1] * test_preds_xgb +
        blend_weights[2] * test_preds_cb
    )
    
    print(f"  Test predictions shape: {test_blended.shape}")
    print(f"  Test predictions range: [{test_blended.min():.4f}, {test_blended.max():.4f}]")
    
    return test_blended


def create_submission(test: pd.DataFrame, predictions: np.ndarray,
                      filename: str = 'submission.csv') -> Path:
    """
    Create submission file.
    """
    submission = pd.DataFrame({
        'user_hash': test['user_hash'],
        'prediction': predictions
    })
    
    # Ensure no NaN
    assert not submission['prediction'].isna().any(), "NaN in predictions!"
    
    # Ensure predictions in [0, 1]
    submission['prediction'] = submission['prediction'].clip(0, 1)
    
    # Save
    submission_path = DATA_DIR / filename
    submission.to_csv(submission_path, index=False)
    
    # Compute checksum
    with open(submission_path, 'rb') as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    
    print(f"\nSubmission saved to: {submission_path}")
    print(f"  Rows: {len(submission):,}")
    print(f"  MD5: {checksum}")
    
    return submission_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Main training pipeline."""
    print("="*60)
    print("MERCOR CHEATING DETECTION - TRAINING PIPELINE")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print(f"N_FOLDS: {N_FOLDS}")
    print("="*60)
    
    # Load data
    train, test, social_graph = load_data()
    
    # Prepare features
    train, test, feature_cols = prepare_features(train, test, social_graph)
    
    # Save feature columns for reproducibility
    with open(ARTIFACTS_DIR / 'feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)
    
    # Quick baseline CV (no optimization)
    print("\n" + "="*60)
    print("BASELINE CROSS-VALIDATION")
    print("="*60)
    
    oof_baseline, results_baseline = cross_validate_with_cost(
        train, feature_cols,
        pseudo_weight=0.05,
        verbose=True
    )
    
    # Save baseline OOF predictions
    oof_df = pd.DataFrame({
        'user_hash': results_baseline['labeled_user_hashes'],
        'prediction': oof_baseline,
        'actual': results_baseline['y_labeled']
    })
    oof_df.to_csv(ARTIFACTS_DIR / 'oof_baseline.csv', index=False)
    
    # Generate test predictions with baseline
    test_preds_baseline = predict_test(test, feature_cols, results_baseline)
    
    # Create submission
    submission_path = create_submission(test, test_preds_baseline, 'submission.csv')
    
    # Save results
    results_to_save = {
        'cv_cost': int(results_baseline['cv_cost']),
        'cv_score': int(results_baseline['cv_score']),
        'mean_fold_cost': float(results_baseline['mean_fold_cost']),
        'std_fold_cost': float(results_baseline['std_fold_cost']),
        't_low': float(results_baseline['t_low']),
        't_high': float(results_baseline['t_high']),
        'blend_weights': [float(w) for w in results_baseline['blend_weights']],
        'pseudo_weight': float(results_baseline['pseudo_weight']),
        'n_folds': N_FOLDS,
        'seed': SEED,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(ARTIFACTS_DIR / 'cv_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save models
    with open(ARTIFACTS_DIR / 'models.pkl', 'wb') as f:
        pickle.dump({
            'models_lgb': results_baseline['models_lgb'],
            'models_xgb': results_baseline['models_xgb'],
            'models_cb': results_baseline['models_cb'],
            'blend_weights': results_baseline['blend_weights']
        }, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"CV Cost: ${results_baseline['cv_cost']:,.0f}")
    print(f"CV Score (LB format): {results_baseline['cv_score']:,.0f}")
    print(f"Optimal thresholds: t_low={results_baseline['t_low']:.4f}, t_high={results_baseline['t_high']:.4f}")
    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")
    print(f"Submission: {submission_path}")
    print("="*60)
    
    return results_baseline


if __name__ == '__main__':
    results = main()
