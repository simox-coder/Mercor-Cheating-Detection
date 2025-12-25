"""
Mercor Cheating Detection - Training Pipeline

This implements:
- Leakage-safe CV on LABELED rows only
- Tabular features + graph features
- LightGBM, XGBoost, CatBoost models
- Ensemble blending
- Optimizes for the official cost metric

Outputs:
- artifacts/oof.csv
- artifacts/cv_summary.json
- submission.csv
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import metric

warnings.filterwarnings('ignore')

# Configuration
SEED = 42
N_FOLDS = 5
N_THRESHOLDS = 200  # Finer threshold search for better optimization

# Directories
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_data(data_dir: str = "."):
    """Load all competition data."""
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    graph = pd.read_csv(os.path.join(data_dir, "social_graph.csv"))
    sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    
    print(f"Train: {len(train):,} rows")
    print(f"Test: {len(test):,} rows")
    print(f"Graph edges: {len(graph):,}")
    
    return train, test, graph, sample_sub


def split_labeled_unlabeled(train: pd.DataFrame):
    """Split train into labeled and unlabeled sets."""
    labeled = train[train['is_cheating'].notna()].copy()
    unlabeled = train[train['is_cheating'].isna()].copy()
    
    labeled['is_cheating'] = labeled['is_cheating'].astype(int)
    
    print(f"\nLabeled: {len(labeled):,} rows ({100*len(labeled)/len(train):.1f}%)")
    print(f"  Cheaters: {(labeled['is_cheating'] == 1).sum():,} ({100*(labeled['is_cheating'] == 1).mean():.1f}%)")
    print(f"  Non-cheaters: {(labeled['is_cheating'] == 0).sum():,}")
    print(f"Unlabeled: {len(unlabeled):,} rows (all high_conf_clean=1)")
    
    return labeled, unlabeled


def create_tabular_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create tabular features including missingness flags and row stats."""
    feature_cols = [f"feature_{i:03d}" for i in range(1, 19)]
    
    features = df[feature_cols].copy()
    
    # Missingness flags
    for col in feature_cols:
        features[f"{col}_missing"] = features[col].isna().astype(int)
    
    # Row statistics (computed on original features)
    features['na_count'] = df[feature_cols].isna().sum(axis=1)
    features['row_mean'] = df[feature_cols].mean(axis=1)
    features['row_std'] = df[feature_cols].std(axis=1)
    features['row_min'] = df[feature_cols].min(axis=1)
    features['row_max'] = df[feature_cols].max(axis=1)
    features['row_range'] = features['row_max'] - features['row_min']
    
    # Fill NaN with -999 for tree models
    features = features.fillna(-999)
    
    feature_names = list(features.columns)
    
    return features, feature_names


def build_graph_structure(graph: pd.DataFrame, all_users: set):
    """
    Build graph adjacency structure.
    
    Returns:
        neighbors: dict mapping user_hash to list of neighbor user_hashes
        degree: dict mapping user_hash to degree
    """
    neighbors = {}
    
    # Add all users (including those not in graph)
    for user in all_users:
        neighbors[user] = []
    
    # Build adjacency list
    for _, row in graph.iterrows():
        user_a, user_b = row['user_a'], row['user_b']
        if user_a not in neighbors:
            neighbors[user_a] = []
        if user_b not in neighbors:
            neighbors[user_b] = []
        neighbors[user_a].append(user_b)
        neighbors[user_b].append(user_a)
    
    # Compute degrees
    degree = {u: len(n) for u, n in neighbors.items()}
    
    return neighbors, degree


def create_graph_features(
    df: pd.DataFrame,
    neighbors: Dict[str, List[str]],
    degree: Dict[str, int],
    train_labels: Optional[Dict[str, int]] = None,
    fold_safe: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create graph-based features.
    
    Args:
        df: DataFrame with user_hash column
        neighbors: Adjacency list
        degree: Degree dictionary
        train_labels: Dict of user_hash -> is_cheating for TRAIN FOLD ONLY (fold-safe)
        fold_safe: Whether to use only train fold labels for neighbor aggregation
        
    Returns:
        graph_features: DataFrame with graph features
        feature_names: List of feature names
    """
    features = pd.DataFrame(index=df.index)
    
    # Basic graph features (computed on full graph, safe)
    features['degree'] = df['user_hash'].map(degree).fillna(0).astype(int)
    features['log_degree'] = np.log1p(features['degree'])
    
    # Neighbor degree statistics (safe, no label info)
    neighbor_degrees = []
    for user in df['user_hash']:
        if user in neighbors and neighbors[user]:
            ndeg = [degree.get(n, 0) for n in neighbors[user]]
            neighbor_degrees.append({
                'mean': np.mean(ndeg),
                'std': np.std(ndeg),
                'min': np.min(ndeg),
                'max': np.max(ndeg),
            })
        else:
            neighbor_degrees.append({'mean': 0, 'std': 0, 'min': 0, 'max': 0})
    
    ndeg_df = pd.DataFrame(neighbor_degrees, index=df.index)
    features['neighbor_degree_mean'] = ndeg_df['mean']
    features['neighbor_degree_std'] = ndeg_df['std']
    features['neighbor_degree_min'] = ndeg_df['min']
    features['neighbor_degree_max'] = ndeg_df['max']
    
    # Neighbor label features (FOLD-SAFE: uses only train fold labels)
    if train_labels is not None and fold_safe:
        n_labeled_neighbors = []
        train_cheat_rates = []
        
        for user in df['user_hash']:
            if user in neighbors and neighbors[user]:
                labeled = [(n, train_labels.get(n)) for n in neighbors[user] if n in train_labels]
                n_labeled = len(labeled)
                if n_labeled > 0:
                    cheat_sum = sum(lab for _, lab in labeled)
                    cheat_rate = cheat_sum / n_labeled
                else:
                    cheat_rate = -1  # Unknown
            else:
                n_labeled = 0
                cheat_rate = -1
            
            n_labeled_neighbors.append(n_labeled)
            train_cheat_rates.append(cheat_rate)
        
        features['n_labeled_neighbors'] = n_labeled_neighbors
        features['train_neighbor_cheat_rate'] = train_cheat_rates
    
    feature_names = list(features.columns)
    
    return features, feature_names


def train_lightgbm(X_train, y_train, X_val, y_val, seed=42):
    """Train LightGBM model."""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': seed,
        'n_jobs': -1,
    }
    
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val, seed=42):
    """Train XGBoost model."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'seed': seed,
        'n_jobs': -1,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    return model


def train_catboost(X_train, y_train, X_val, y_val, seed=42):
    """Train CatBoost model."""
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=seed,
        verbose=False,
        early_stopping_rounds=50,
        task_type='CPU',
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    return model


def predict_lightgbm(model, X):
    """Get LightGBM predictions."""
    return model.predict(X)


def predict_xgboost(model, X):
    """Get XGBoost predictions."""
    return model.predict(xgb.DMatrix(X))


def predict_catboost(model, X):
    """Get CatBoost predictions."""
    return model.predict_proba(X)[:, 1]


def run_cv(
    labeled_df: pd.DataFrame,
    neighbors: Dict[str, List[str]],
    degree: Dict[str, int],
    n_folds: int = N_FOLDS,
    seed: int = SEED
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run cross-validation with leakage-safe graph features.
    
    Returns:
        oof_df: DataFrame with OOF predictions
        cv_summary: Dictionary with CV summary statistics
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION")
    print("="*60)
    
    # Prepare data
    y = labeled_df['is_cheating'].values
    user_hashes = labeled_df['user_hash'].values
    
    # Create tabular features
    tab_features, tab_feature_names = create_tabular_features(labeled_df)
    print(f"Tabular features: {len(tab_feature_names)}")
    
    # Initialize OOF arrays
    n_samples = len(labeled_df)
    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cat = np.zeros(n_samples)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(labeled_df, y)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,}")
        
        # Get train fold labels for fold-safe graph features
        train_users = user_hashes[train_idx]
        train_labels_dict = dict(zip(train_users, y[train_idx]))
        
        # Create graph features (fold-safe)
        graph_features, graph_feature_names = create_graph_features(
            labeled_df, neighbors, degree, train_labels_dict, fold_safe=True
        )
        
        # Combine features
        X = pd.concat([tab_features, graph_features], axis=1)
        feature_names = tab_feature_names + graph_feature_names
        print(f"  Total features: {len(feature_names)}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train models
        print("  Training LightGBM...")
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, seed=seed)
        oof_lgb[val_idx] = predict_lightgbm(lgb_model, X_val)
        
        print("  Training XGBoost...")
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, seed=seed)
        oof_xgb[val_idx] = predict_xgboost(xgb_model, X_val)
        
        print("  Training CatBoost...")
        cat_model = train_catboost(X_train, y_train, X_val, y_val, seed=seed)
        oof_cat[val_idx] = predict_catboost(cat_model, X_val)
        
        # Evaluate fold
        for name, preds in [('LGB', oof_lgb[val_idx]), ('XGB', oof_xgb[val_idx]), ('CAT', oof_cat[val_idx])]:
            result = metric.evaluate(y_val, preds, n_thresholds=N_THRESHOLDS)
            print(f"    {name}: cost={result.best_cost:,}, t_low={result.t_low:.3f}, t_high={result.t_high:.3f}")
        
        # Blend predictions
        blend = (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx]) / 3
        result = metric.evaluate(y_val, blend, n_thresholds=N_THRESHOLDS)
        print(f"    BLEND: cost={result.best_cost:,}, t_low={result.t_low:.3f}, t_high={result.t_high:.3f}")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'cost_lgb': int(metric.evaluate(y_val, oof_lgb[val_idx], n_thresholds=N_THRESHOLDS).best_cost),
            'cost_xgb': int(metric.evaluate(y_val, oof_xgb[val_idx], n_thresholds=N_THRESHOLDS).best_cost),
            'cost_cat': int(metric.evaluate(y_val, oof_cat[val_idx], n_thresholds=N_THRESHOLDS).best_cost),
            'cost_blend': int(result.best_cost),
        })
    
    # Create OOF DataFrame
    oof_df = pd.DataFrame({
        'user_hash': user_hashes,
        'is_cheating': y,
        'pred_lgb': oof_lgb,
        'pred_xgb': oof_xgb,
        'pred_cat': oof_cat,
        'pred_blend': (oof_lgb + oof_xgb + oof_cat) / 3,
    })
    
    # Final OOF evaluation
    print("\n" + "="*60)
    print("OOF EVALUATION")
    print("="*60)
    
    cv_results = {}
    for name, col in [('LGB', 'pred_lgb'), ('XGB', 'pred_xgb'), ('CAT', 'pred_cat'), ('BLEND', 'pred_blend')]:
        result = metric.evaluate(y, oof_df[col].values, n_thresholds=N_THRESHOLDS)
        cv_results[name] = {
            'cost': result.best_cost,
            'score': result.best_score,
            't_low': result.t_low,
            't_high': result.t_high,
            'n_auto_pass': result.n_auto_pass,
            'n_manual_review': result.n_manual_review,
            'n_auto_block': result.n_auto_block,
            'breakdown': result.breakdown,
        }
        print(f"\n{name}:")
        print(f"  Cost: {result.best_cost:,} | Score: {result.best_score:,}")
        print(f"  Thresholds: t_low={result.t_low:.4f}, t_high={result.t_high:.4f}")
        print(f"  Regions: pass={result.n_auto_pass:,}, review={result.n_manual_review:,}, block={result.n_auto_block:,}")
    
    # CV summary
    cv_summary = {
        'n_folds': n_folds,
        'seed': seed,
        'n_samples': n_samples,
        'n_cheaters': int(y.sum()),
        'fold_results': fold_results,
        'oof_results': cv_results,
        'best_model': min(cv_results.keys(), key=lambda k: cv_results[k]['cost']),
        'timestamp': datetime.now().isoformat(),
    }
    
    return oof_df, cv_summary


def train_final_models(
    labeled_df: pd.DataFrame,
    neighbors: Dict[str, List[str]],
    degree: Dict[str, int],
    seed: int = SEED
):
    """Train final models on all labeled data."""
    print("\n" + "="*60)
    print("TRAINING FINAL MODELS")
    print("="*60)
    
    y = labeled_df['is_cheating'].values
    
    # Create features
    tab_features, _ = create_tabular_features(labeled_df)
    
    # Use all labeled data for graph features
    all_labels_dict = dict(zip(labeled_df['user_hash'], y))
    graph_features, _ = create_graph_features(
        labeled_df, neighbors, degree, all_labels_dict, fold_safe=True
    )
    
    X = pd.concat([tab_features, graph_features], axis=1)
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {len(X):,}")
    
    # Train models
    print("\nTraining LightGBM...")
    lgb_model = train_lightgbm(X, y, X, y, seed=seed)
    
    print("Training XGBoost...")
    xgb_model = train_xgboost(X, y, X, y, seed=seed)
    
    print("Training CatBoost...")
    cat_model = train_catboost(X, y, X, y, seed=seed)
    
    return {
        'lgb': lgb_model,
        'xgb': xgb_model,
        'cat': cat_model,
    }, X.columns.tolist()


def create_test_predictions(
    test_df: pd.DataFrame,
    models: Dict,
    feature_names: List[str],
    neighbors: Dict[str, List[str]],
    degree: Dict[str, int],
    labeled_labels: Dict[str, int]
):
    """Create predictions for test set."""
    print("\n" + "="*60)
    print("CREATING TEST PREDICTIONS")
    print("="*60)
    
    # Create features
    tab_features, _ = create_tabular_features(test_df)
    graph_features, _ = create_graph_features(
        test_df, neighbors, degree, labeled_labels, fold_safe=True
    )
    
    X_test = pd.concat([tab_features, graph_features], axis=1)
    
    # Ensure same feature order
    X_test = X_test[feature_names]
    
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_test.shape[1]}")
    
    # Predict
    pred_lgb = predict_lightgbm(models['lgb'], X_test)
    pred_xgb = predict_xgboost(models['xgb'], X_test)
    pred_cat = predict_catboost(models['cat'], X_test)
    
    # Blend
    pred_blend = (pred_lgb + pred_xgb + pred_cat) / 3
    
    print(f"\nPrediction ranges:")
    print(f"  LGB: [{pred_lgb.min():.4f}, {pred_lgb.max():.4f}]")
    print(f"  XGB: [{pred_xgb.min():.4f}, {pred_xgb.max():.4f}]")
    print(f"  CAT: [{pred_cat.min():.4f}, {pred_cat.max():.4f}]")
    print(f"  BLEND: [{pred_blend.min():.4f}, {pred_blend.max():.4f}]")
    
    return {
        'lgb': pred_lgb,
        'xgb': pred_xgb,
        'cat': pred_cat,
        'blend': pred_blend,
    }


def create_submission(test_df: pd.DataFrame, predictions: np.ndarray, 
                      sample_sub: pd.DataFrame, output_path: str = "submission.csv"):
    """Create submission file."""
    submission = pd.DataFrame({
        'user_hash': test_df['user_hash'],
        'prediction': predictions
    })
    
    # Ensure same order as sample submission
    submission = sample_sub[['user_hash']].merge(submission, on='user_hash', how='left')
    
    # Validate
    assert len(submission) == len(sample_sub), f"Row count mismatch: {len(submission)} vs {len(sample_sub)}"
    assert submission['prediction'].notna().all(), "Submission contains NaN"
    assert submission['prediction'].min() >= 0, f"Predictions below 0: {submission['prediction'].min()}"
    assert submission['prediction'].max() <= 1, f"Predictions above 1: {submission['prediction'].max()}"
    
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"  Rows: {len(submission):,}")
    print(f"  Predictions: min={submission['prediction'].min():.4f}, max={submission['prediction'].max():.4f}")
    
    return submission


def main():
    print("="*60)
    print("MERCOR CHEATING DETECTION - TRAINING PIPELINE")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load data
    train, test, graph, sample_sub = load_data(".")
    
    # Split labeled/unlabeled
    labeled, unlabeled = split_labeled_unlabeled(train)
    
    # Build graph structure
    print("\nBuilding graph structure...")
    all_users = set(train['user_hash'].tolist() + test['user_hash'].tolist())
    neighbors, degree = build_graph_structure(graph, all_users)
    print(f"  Nodes in graph: {len(neighbors):,}")
    print(f"  Max degree: {max(degree.values())}")
    
    # Run cross-validation
    oof_df, cv_summary = run_cv(labeled, neighbors, degree)
    
    # Save OOF predictions
    oof_path = os.path.join(ARTIFACTS_DIR, "oof.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"\nOOF predictions saved to: {oof_path}")
    
    # Save CV summary
    cv_summary_path = os.path.join(ARTIFACTS_DIR, "cv_summary.json")
    with open(cv_summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2, default=str)
    print(f"CV summary saved to: {cv_summary_path}")
    
    # Train final models
    models, feature_names = train_final_models(labeled, neighbors, degree)
    
    # Create test predictions
    labeled_labels = dict(zip(labeled['user_hash'], labeled['is_cheating']))
    test_preds = create_test_predictions(test, models, feature_names, neighbors, degree, labeled_labels)
    
    # Create submission with blend
    submission = create_submission(test, test_preds['blend'], sample_sub, "submission.csv")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    best_model = cv_summary['best_model']
    best_result = cv_summary['oof_results'][best_model]
    
    print(f"Best model: {best_model}")
    print(f"CV Cost: {best_result['cost']:,}")
    print(f"CV Score: {best_result['score']:,}")
    print(f"Thresholds: t_low={best_result['t_low']:.4f}, t_high={best_result['t_high']:.4f}")
    print(f"Regions: pass={best_result['n_auto_pass']:,}, review={best_result['n_manual_review']:,}, block={best_result['n_auto_block']:,}")
    
    # Cost breakdown
    print("\nCost breakdown:")
    for key in ['fn_auto_pass', 'fp_auto_block', 'fp_manual', 'tp_manual']:
        count = best_result['breakdown'].get(key, 0)
        cost_key = f"cost_{key}"
        cost = best_result['breakdown'].get(cost_key, 0)
        print(f"  {key}: {count:,} (cost: {cost:,})")
    
    print("\n" + "="*60)
    print("COMMANDS TO REPRODUCE")
    print("="*60)
    print("pip install pandas numpy scikit-learn lightgbm xgboost catboost")
    print("python metric_tests.py")
    print("python sanity_checks.py")
    print("python train.py")
    
    return cv_summary


if __name__ == "__main__":
    cv_summary = main()
