"""
Mercor Cheating Detection - Advanced Training Pipeline

This version includes:
- More sophisticated graph features
- Improved hyperparameters
- Rank-based blending
- Better feature engineering
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
from scipy.stats import rankdata

import metric

warnings.filterwarnings('ignore')

# Configuration
SEED = 42
N_FOLDS = 5
N_THRESHOLDS = 300  # Finer threshold search

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
    """Create extensive tabular features."""
    feature_cols = [f"feature_{i:03d}" for i in range(1, 19)]
    
    features = df[feature_cols].copy()
    
    # Missingness flags
    for col in feature_cols:
        features[f"{col}_missing"] = features[col].isna().astype(int)
    
    # Row statistics
    features['na_count'] = df[feature_cols].isna().sum(axis=1)
    features['row_mean'] = df[feature_cols].mean(axis=1)
    features['row_std'] = df[feature_cols].std(axis=1)
    features['row_min'] = df[feature_cols].min(axis=1)
    features['row_max'] = df[feature_cols].max(axis=1)
    features['row_range'] = features['row_max'] - features['row_min']
    features['row_sum'] = df[feature_cols].sum(axis=1)
    features['row_median'] = df[feature_cols].median(axis=1)
    
    # Feature interactions
    # Normalize feature_015 (wide range)
    f15 = df['feature_015'].fillna(0)
    features['feature_015_log'] = np.log1p(np.abs(f15)) * np.sign(f15)
    features['feature_015_bin'] = pd.cut(f15, bins=[-np.inf, 0, 1, 10, 100, np.inf], 
                                          labels=[0, 1, 2, 3, 4]).astype(float)
    
    # feature_010 has wide range too
    f10 = df['feature_010'].fillna(0)
    features['feature_010_log'] = np.log1p(f10)
    features['feature_010_bin'] = (f10 > 0).astype(int)
    
    # Binary feature combinations
    for i in [7, 11, 13, 14]:
        col = f'feature_{i:03d}'
        features[f'{col}_x_na_count'] = df[col].fillna(0) * features['na_count']
    
    # Ratios
    features['ratio_018_017'] = df['feature_018'].fillna(0) / (df['feature_017'].fillna(0) + 1e-6)
    features['ratio_high_to_low'] = (df['feature_010'].fillna(0) + 1) / (df['feature_015'].fillna(0).abs() + 1)
    
    # Fill NaN with -999 for tree models
    features = features.fillna(-999)
    
    feature_names = list(features.columns)
    
    return features, feature_names


def build_graph_structure(graph: pd.DataFrame, all_users: set):
    """Build comprehensive graph adjacency structure."""
    neighbors = {u: [] for u in all_users}
    
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
    
    # Find connected components using iterative Union-Find
    parent = {u: u for u in neighbors}
    rank = {u: 0 for u in neighbors}
    
    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        # Union by rank
        if rank[px] < rank[py]:
            parent[px] = py
        elif rank[px] > rank[py]:
            parent[py] = px
        else:
            parent[py] = px
            rank[px] += 1
    
    for _, row in graph.iterrows():
        union(row['user_a'], row['user_b'])
    
    # Get component ID for each user
    component = {u: find(u) for u in neighbors}
    
    # Component sizes
    comp_sizes = {}
    for u, c in component.items():
        comp_sizes[c] = comp_sizes.get(c, 0) + 1
    
    user_comp_size = {u: comp_sizes[component[u]] for u in neighbors}
    
    return neighbors, degree, component, user_comp_size


def create_graph_features(
    df: pd.DataFrame,
    neighbors: Dict[str, List[str]],
    degree: Dict[str, int],
    user_comp_size: Dict[str, int],
    train_labels: Optional[Dict[str, int]] = None,
    train_preds: Optional[Dict[str, float]] = None,
    fold_safe: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """Create comprehensive graph-based features."""
    features = pd.DataFrame(index=df.index)
    
    # Basic graph features
    features['degree'] = df['user_hash'].map(degree).fillna(0).astype(int)
    features['log_degree'] = np.log1p(features['degree'])
    features['degree_sq'] = features['degree'] ** 0.5
    features['comp_size'] = df['user_hash'].map(user_comp_size).fillna(1).astype(int)
    features['log_comp_size'] = np.log1p(features['comp_size'])
    features['is_isolated'] = (features['degree'] == 0).astype(int)
    
    # Neighbor degree statistics
    neighbor_stats = []
    for user in df['user_hash']:
        if user in neighbors and neighbors[user]:
            ndeg = [degree.get(n, 0) for n in neighbors[user]]
            neighbor_stats.append({
                'mean': np.mean(ndeg),
                'std': np.std(ndeg) if len(ndeg) > 1 else 0,
                'min': np.min(ndeg),
                'max': np.max(ndeg),
                'sum': np.sum(ndeg),
                'median': np.median(ndeg),
            })
        else:
            neighbor_stats.append({'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'sum': 0, 'median': 0})
    
    ndeg_df = pd.DataFrame(neighbor_stats, index=df.index)
    features['neighbor_degree_mean'] = ndeg_df['mean']
    features['neighbor_degree_std'] = ndeg_df['std']
    features['neighbor_degree_min'] = ndeg_df['min']
    features['neighbor_degree_max'] = ndeg_df['max']
    features['neighbor_degree_sum'] = ndeg_df['sum']
    features['neighbor_degree_median'] = ndeg_df['median']
    
    # Neighbor label features (FOLD-SAFE)
    if train_labels is not None and fold_safe:
        label_stats = []
        for user in df['user_hash']:
            if user in neighbors and neighbors[user]:
                labeled = [(n, train_labels.get(n)) for n in neighbors[user] if n in train_labels]
                n_labeled = len(labeled)
                if n_labeled > 0:
                    cheat_sum = sum(lab for _, lab in labeled)
                    cheat_rate = cheat_sum / n_labeled
                else:
                    cheat_rate = -1
            else:
                n_labeled = 0
                cheat_rate = -1
            
            label_stats.append({
                'n_labeled': n_labeled,
                'cheat_rate': cheat_rate,
            })
        
        lstat_df = pd.DataFrame(label_stats, index=df.index)
        features['n_labeled_neighbors'] = lstat_df['n_labeled']
        features['train_neighbor_cheat_rate'] = lstat_df['cheat_rate']
        features['labeled_ratio'] = lstat_df['n_labeled'] / (features['degree'] + 1)
    
    # OOF prediction propagation (FOLD-SAFE)
    if train_preds is not None:
        pred_stats = []
        for user in df['user_hash']:
            if user in neighbors and neighbors[user]:
                npreds = [train_preds.get(n) for n in neighbors[user] if n in train_preds]
                if npreds:
                    pred_stats.append({
                        'mean': np.mean(npreds),
                        'std': np.std(npreds) if len(npreds) > 1 else 0,
                        'max': np.max(npreds),
                        'min': np.min(npreds),
                    })
                else:
                    pred_stats.append({'mean': -1, 'std': 0, 'max': -1, 'min': -1})
            else:
                pred_stats.append({'mean': -1, 'std': 0, 'max': -1, 'min': -1})
        
        pstat_df = pd.DataFrame(pred_stats, index=df.index)
        features['neighbor_pred_mean'] = pstat_df['mean']
        features['neighbor_pred_std'] = pstat_df['std']
        features['neighbor_pred_max'] = pstat_df['max']
        features['neighbor_pred_min'] = pstat_df['min']
    
    feature_names = list(features.columns)
    
    return features, feature_names


def train_lightgbm(X_train, y_train, X_val, y_val, seed=42, params=None):
    """Train LightGBM with optimized parameters."""
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'verbose': -1,
            'seed': seed,
            'n_jobs': -1,
        }
    
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val, seed=42, params=None):
    """Train XGBoost with optimized parameters."""
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'max_depth': 7,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 10,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'seed': seed,
            'n_jobs': -1,
        }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
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


def train_catboost(X_train, y_train, X_val, y_val, seed=42):
    """Train CatBoost with optimized parameters."""
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5,
        random_seed=seed,
        verbose=False,
        early_stopping_rounds=100,
        task_type='CPU',
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    return model


def predict_lightgbm(model, X):
    return model.predict(X)


def predict_xgboost(model, X):
    return model.predict(xgb.DMatrix(X))


def predict_catboost(model, X):
    return model.predict_proba(X)[:, 1]


def rank_average(predictions: List[np.ndarray]) -> np.ndarray:
    """Compute rank-averaged predictions."""
    n = len(predictions[0])
    ranks = [rankdata(p) / n for p in predictions]
    return np.mean(ranks, axis=0)


def run_cv_advanced(
    labeled_df: pd.DataFrame,
    neighbors: Dict[str, List[str]],
    degree: Dict[str, int],
    user_comp_size: Dict[str, int],
    n_folds: int = N_FOLDS,
    seed: int = SEED
) -> Tuple[pd.DataFrame, Dict]:
    """Run advanced cross-validation with improved features."""
    print("\n" + "="*60)
    print("ADVANCED CROSS-VALIDATION")
    print("="*60)
    
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
        
        # Get train fold labels
        train_users = user_hashes[train_idx]
        train_labels_dict = dict(zip(train_users, y[train_idx]))
        
        # Create graph features (fold-safe)
        graph_features, graph_feature_names = create_graph_features(
            labeled_df, neighbors, degree, user_comp_size,
            train_labels_dict, None, fold_safe=True
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
        
        # Blends
        prob_blend = (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx]) / 3
        rank_blend = rank_average([oof_lgb[val_idx], oof_xgb[val_idx], oof_cat[val_idx]])
        
        result_prob = metric.evaluate(y_val, prob_blend, n_thresholds=N_THRESHOLDS)
        result_rank = metric.evaluate(y_val, rank_blend, n_thresholds=N_THRESHOLDS)
        
        print(f"    PROB_BLEND: cost={result_prob.best_cost:,}")
        print(f"    RANK_BLEND: cost={result_rank.best_cost:,}")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'cost_lgb': int(metric.evaluate(y_val, oof_lgb[val_idx], n_thresholds=N_THRESHOLDS).best_cost),
            'cost_xgb': int(metric.evaluate(y_val, oof_xgb[val_idx], n_thresholds=N_THRESHOLDS).best_cost),
            'cost_cat': int(metric.evaluate(y_val, oof_cat[val_idx], n_thresholds=N_THRESHOLDS).best_cost),
            'cost_prob_blend': int(result_prob.best_cost),
            'cost_rank_blend': int(result_rank.best_cost),
        })
    
    # Create OOF DataFrame
    prob_blend_full = (oof_lgb + oof_xgb + oof_cat) / 3
    rank_blend_full = rank_average([oof_lgb, oof_xgb, oof_cat])
    
    oof_df = pd.DataFrame({
        'user_hash': user_hashes,
        'is_cheating': y,
        'pred_lgb': oof_lgb,
        'pred_xgb': oof_xgb,
        'pred_cat': oof_cat,
        'pred_prob_blend': prob_blend_full,
        'pred_rank_blend': rank_blend_full,
    })
    
    # Final OOF evaluation
    print("\n" + "="*60)
    print("OOF EVALUATION")
    print("="*60)
    
    cv_results = {}
    for name, col in [('LGB', 'pred_lgb'), ('XGB', 'pred_xgb'), ('CAT', 'pred_cat'), 
                      ('PROB_BLEND', 'pred_prob_blend'), ('RANK_BLEND', 'pred_rank_blend')]:
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


def train_final_models_advanced(
    labeled_df: pd.DataFrame,
    neighbors: Dict[str, List[str]],
    degree: Dict[str, int],
    user_comp_size: Dict[str, int],
    oof_preds: Dict[str, float],
    seed: int = SEED
):
    """Train final models with all features."""
    print("\n" + "="*60)
    print("TRAINING FINAL MODELS")
    print("="*60)
    
    y = labeled_df['is_cheating'].values
    
    # Create features
    tab_features, tab_names = create_tabular_features(labeled_df)
    
    all_labels_dict = dict(zip(labeled_df['user_hash'], y))
    graph_features, graph_names = create_graph_features(
        labeled_df, neighbors, degree, user_comp_size,
        all_labels_dict, oof_preds, fold_safe=True
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


def create_test_predictions_advanced(
    test_df: pd.DataFrame,
    models: Dict,
    feature_names: List[str],
    neighbors: Dict[str, List[str]],
    degree: Dict[str, int],
    user_comp_size: Dict[str, int],
    labeled_labels: Dict[str, int],
    labeled_preds: Dict[str, float]
):
    """Create predictions for test set."""
    print("\n" + "="*60)
    print("CREATING TEST PREDICTIONS")
    print("="*60)
    
    # Create features
    tab_features, _ = create_tabular_features(test_df)
    graph_features, _ = create_graph_features(
        test_df, neighbors, degree, user_comp_size,
        labeled_labels, labeled_preds, fold_safe=True
    )
    
    X_test = pd.concat([tab_features, graph_features], axis=1)
    X_test = X_test[feature_names]
    
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_test.shape[1]}")
    
    # Predict
    pred_lgb = predict_lightgbm(models['lgb'], X_test)
    pred_xgb = predict_xgboost(models['xgb'], X_test)
    pred_cat = predict_catboost(models['cat'], X_test)
    
    # Blends
    pred_prob_blend = (pred_lgb + pred_xgb + pred_cat) / 3
    pred_rank_blend = rank_average([pred_lgb, pred_xgb, pred_cat])
    
    print(f"\nPrediction ranges:")
    for name, pred in [('LGB', pred_lgb), ('XGB', pred_xgb), ('CAT', pred_cat), 
                       ('PROB', pred_prob_blend), ('RANK', pred_rank_blend)]:
        print(f"  {name}: [{pred.min():.4f}, {pred.max():.4f}]")
    
    return {
        'lgb': pred_lgb,
        'xgb': pred_xgb,
        'cat': pred_cat,
        'prob_blend': pred_prob_blend,
        'rank_blend': pred_rank_blend,
    }


def create_submission(test_df: pd.DataFrame, predictions: np.ndarray, 
                      sample_sub: pd.DataFrame, output_path: str = "submission.csv"):
    """Create submission file."""
    submission = pd.DataFrame({
        'user_hash': test_df['user_hash'],
        'prediction': predictions
    })
    
    submission = sample_sub[['user_hash']].merge(submission, on='user_hash', how='left')
    
    assert len(submission) == len(sample_sub)
    assert submission['prediction'].notna().all()
    assert submission['prediction'].min() >= 0
    assert submission['prediction'].max() <= 1
    
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"  Rows: {len(submission):,}")
    print(f"  Predictions: min={submission['prediction'].min():.4f}, max={submission['prediction'].max():.4f}")
    
    return submission


def main():
    print("="*60)
    print("MERCOR CHEATING DETECTION - ADVANCED PIPELINE")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load data
    train, test, graph, sample_sub = load_data(".")
    
    # Split labeled/unlabeled
    labeled, unlabeled = split_labeled_unlabeled(train)
    
    # Build graph structure
    print("\nBuilding graph structure...")
    all_users = set(train['user_hash'].tolist() + test['user_hash'].tolist())
    neighbors, degree, component, user_comp_size = build_graph_structure(graph, all_users)
    print(f"  Nodes in graph: {len(neighbors):,}")
    print(f"  Max degree: {max(degree.values())}")
    print(f"  Components: {len(set(component.values())):,}")
    
    # Run cross-validation
    oof_df, cv_summary = run_cv_advanced(labeled, neighbors, degree, user_comp_size)
    
    # Save OOF predictions
    oof_path = os.path.join(ARTIFACTS_DIR, "oof_advanced.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"\nOOF predictions saved to: {oof_path}")
    
    # Save CV summary
    cv_summary_path = os.path.join(ARTIFACTS_DIR, "cv_summary_advanced.json")
    with open(cv_summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2, default=str)
    
    # Prepare OOF predictions dict for neighbor propagation
    best_col = 'pred_prob_blend' if cv_summary['best_model'] == 'PROB_BLEND' else 'pred_rank_blend'
    oof_preds_dict = dict(zip(oof_df['user_hash'], oof_df[best_col]))
    
    # Train final models
    models, feature_names = train_final_models_advanced(
        labeled, neighbors, degree, user_comp_size, oof_preds_dict
    )
    
    # Create test predictions
    labeled_labels = dict(zip(labeled['user_hash'], labeled['is_cheating']))
    test_preds = create_test_predictions_advanced(
        test, models, feature_names, neighbors, degree, user_comp_size,
        labeled_labels, oof_preds_dict
    )
    
    # Determine best blend
    best_pred_key = 'prob_blend' if cv_summary['best_model'] == 'PROB_BLEND' else 'rank_blend'
    submission = create_submission(test, test_preds[best_pred_key], sample_sub, "submission_advanced.csv")
    
    # Also create prob blend submission
    create_submission(test, test_preds['prob_blend'], sample_sub, "submission.csv")
    
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
    
    # Per-sample and expected test cost
    test_size = len(test)  # Use actual test size instead of hardcoded
    per_sample_cost = best_result['cost'] / cv_summary['n_samples']
    expected_test_cost = per_sample_cost * test_size
    print(f"\nPer-sample cost: {per_sample_cost:.2f}")
    print(f"Expected test cost (proportional): {expected_test_cost:,.0f}")
    print(f"Expected public LB score: {-expected_test_cost:,.0f}")
    
    return cv_summary


if __name__ == "__main__":
    cv_summary = main()
