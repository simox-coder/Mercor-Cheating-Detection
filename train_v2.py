"""
Mercor Cheating Detection - Improved Training Pipeline v2

Key improvements:
1. Experiment logging with proxy split validation
2. More sophisticated graph features including 2-hop neighbors
3. Pseudo-labeling with high_conf_clean rows
4. More aggressive hyperparameter tuning
5. Multiple blending strategies
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata

import metric
from experiment_framework import ExperimentTracker, log_experiment, save_experiment_artifacts, generate_exp_id

warnings.filterwarnings('ignore')

# Configuration
SEED = 42
N_FOLDS = 5
N_THRESHOLDS = 300

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


def split_data(train: pd.DataFrame):
    """Split train into labeled and unlabeled sets."""
    labeled = train[train['is_cheating'].notna()].copy()
    unlabeled = train[train['is_cheating'].isna()].copy()
    
    labeled['is_cheating'] = labeled['is_cheating'].astype(int)
    
    print(f"\nLabeled: {len(labeled):,} rows ({100*len(labeled)/len(train):.1f}%)")
    print(f"  Cheaters: {(labeled['is_cheating'] == 1).sum():,} ({100*(labeled['is_cheating'] == 1).mean():.1f}%)")
    print(f"  Non-cheaters: {(labeled['is_cheating'] == 0).sum():,}")
    print(f"Unlabeled: {len(unlabeled):,} rows (high_conf_clean=1)")
    
    return labeled, unlabeled


def build_graph(graph_df: pd.DataFrame, all_users: set):
    """Build graph structure with efficient neighbor lookup."""
    neighbors = defaultdict(list)
    
    for _, row in graph_df.iterrows():
        u, v = row['user_a'], row['user_b']
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    # Ensure all users are in neighbors dict
    for u in all_users:
        if u not in neighbors:
            neighbors[u] = []
    
    degree = {u: len(n) for u, n in neighbors.items()}
    
    return dict(neighbors), degree


def get_2hop_neighbors(user: str, neighbors: Dict) -> set:
    """Get 2-hop neighbors for a user."""
    hop1 = set(neighbors.get(user, []))
    hop2 = set()
    for n in hop1:
        hop2.update(neighbors.get(n, []))
    hop2 -= hop1
    hop2.discard(user)
    return hop2


def create_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create tabular features with comprehensive engineering."""
    feature_cols = [f"feature_{i:03d}" for i in range(1, 19)]
    features = df[feature_cols].copy()
    
    # Missingness flags
    for col in feature_cols:
        features[f"{col}_miss"] = features[col].isna().astype(int)
    
    # Row statistics
    features['na_count'] = df[feature_cols].isna().sum(axis=1)
    features['row_mean'] = df[feature_cols].mean(axis=1)
    features['row_std'] = df[feature_cols].std(axis=1)
    features['row_min'] = df[feature_cols].min(axis=1)
    features['row_max'] = df[feature_cols].max(axis=1)
    features['row_range'] = features['row_max'] - features['row_min']
    features['row_sum'] = df[feature_cols].sum(axis=1)
    features['row_median'] = df[feature_cols].median(axis=1)
    features['row_skew'] = df[feature_cols].skew(axis=1)
    features['row_kurt'] = df[feature_cols].kurt(axis=1)
    
    # Feature transformations
    for col in ['feature_010', 'feature_015', 'feature_016']:
        val = df[col].fillna(0)
        features[f'{col}_log'] = np.log1p(np.abs(val)) * np.sign(val)
        features[f'{col}_sqrt'] = np.sqrt(np.abs(val)) * np.sign(val)
    
    # Binning for wide-range features
    f15 = df['feature_015'].fillna(0)
    features['f015_bin'] = pd.cut(f15, bins=[-np.inf, 0, 1, 10, 50, 100, np.inf], 
                                  labels=[0, 1, 2, 3, 4, 5]).astype(float)
    
    f10 = df['feature_010'].fillna(0)
    features['f010_bin'] = pd.cut(f10, bins=[-np.inf, 0, 100, 500, 1000, 5000, np.inf],
                                  labels=[0, 1, 2, 3, 4, 5]).astype(float)
    features['f010_nonzero'] = (f10 > 0).astype(int)
    
    # Binary feature combinations
    binary_cols = ['feature_007', 'feature_011', 'feature_013', 'feature_014']
    for col in binary_cols:
        val = df[col].fillna(0)
        features[f'{col}_x_na'] = val * features['na_count']
    
    # Ratios and interactions
    features['ratio_018_017'] = df['feature_018'].fillna(0) / (df['feature_017'].fillna(0) + 0.01)
    features['ratio_f10_f15'] = (f10 + 1) / (np.abs(f15) + 1)
    features['f016_x_f017'] = df['feature_016'].fillna(0) * df['feature_017'].fillna(0)
    
    # High-value indicators
    features['high_f010'] = (f10 > 1000).astype(int)
    features['high_f015'] = (f15 > 100).astype(int)
    
    # Fill remaining NaN
    features = features.fillna(-999)
    
    return features


def create_graph_features(
    df: pd.DataFrame,
    neighbors: Dict,
    degree: Dict,
    train_labels: Optional[Dict[str, int]] = None,
    train_preds: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Create graph features with 2-hop information."""
    features = pd.DataFrame(index=df.index)
    
    # Basic degree features
    features['degree'] = df['user_hash'].map(degree).fillna(0)
    features['log_degree'] = np.log1p(features['degree'])
    features['sqrt_degree'] = np.sqrt(features['degree'])
    features['is_isolated'] = (features['degree'] == 0).astype(int)
    
    # Neighbor statistics
    neighbor_stats = []
    for user in df['user_hash']:
        ns = neighbors.get(user, [])
        if ns:
            ndeg = [degree.get(n, 0) for n in ns]
            hop2 = get_2hop_neighbors(user, neighbors)
            hop2_deg = [degree.get(n, 0) for n in hop2] if hop2 else [0]
            
            stats = {
                'n_mean_deg': np.mean(ndeg),
                'n_std_deg': np.std(ndeg) if len(ndeg) > 1 else 0,
                'n_min_deg': np.min(ndeg),
                'n_max_deg': np.max(ndeg),
                'n_sum_deg': np.sum(ndeg),
                'n_count': len(ns),
                'hop2_count': len(hop2),
                'hop2_mean_deg': np.mean(hop2_deg),
            }
        else:
            stats = {
                'n_mean_deg': 0, 'n_std_deg': 0, 'n_min_deg': 0, 
                'n_max_deg': 0, 'n_sum_deg': 0, 'n_count': 0,
                'hop2_count': 0, 'hop2_mean_deg': 0,
            }
        neighbor_stats.append(stats)
    
    nstat_df = pd.DataFrame(neighbor_stats, index=df.index)
    for col in nstat_df.columns:
        features[col] = nstat_df[col]
    
    # Fold-safe label features (only use train fold labels)
    if train_labels:
        label_stats = []
        for user in df['user_hash']:
            ns = neighbors.get(user, [])
            if ns:
                labeled_ns = [(n, train_labels.get(n)) for n in ns if n in train_labels]
                n_labeled = len(labeled_ns)
                if n_labeled > 0:
                    cheat_sum = sum(lab for _, lab in labeled_ns)
                    cheat_rate = cheat_sum / n_labeled
                    # Also get hop2 label stats
                    hop2 = get_2hop_neighbors(user, neighbors)
                    labeled_hop2 = [(n, train_labels.get(n)) for n in hop2 if n in train_labels]
                    n_labeled_hop2 = len(labeled_hop2)
                    hop2_cheat_rate = sum(lab for _, lab in labeled_hop2) / n_labeled_hop2 if n_labeled_hop2 > 0 else -1
                else:
                    cheat_rate = -1
                    n_labeled_hop2 = 0
                    hop2_cheat_rate = -1
            else:
                n_labeled = 0
                cheat_rate = -1
                n_labeled_hop2 = 0
                hop2_cheat_rate = -1
            
            label_stats.append({
                'n_labeled': n_labeled,
                'cheat_rate': cheat_rate,
                'n_labeled_hop2': n_labeled_hop2,
                'hop2_cheat_rate': hop2_cheat_rate,
            })
        
        lstat_df = pd.DataFrame(label_stats, index=df.index)
        features['n_labeled_neighbors'] = lstat_df['n_labeled']
        features['neighbor_cheat_rate'] = lstat_df['cheat_rate']
        features['labeled_ratio'] = lstat_df['n_labeled'] / (features['degree'] + 1)
        features['n_labeled_hop2'] = lstat_df['n_labeled_hop2']
        features['hop2_cheat_rate'] = lstat_df['hop2_cheat_rate']
    
    # Fold-safe prediction propagation
    if train_preds:
        pred_stats = []
        for user in df['user_hash']:
            ns = neighbors.get(user, [])
            if ns:
                npreds = [train_preds.get(n) for n in ns if n in train_preds]
                if npreds:
                    pred_stats.append({
                        'pred_mean': np.mean(npreds),
                        'pred_std': np.std(npreds) if len(npreds) > 1 else 0,
                        'pred_max': np.max(npreds),
                        'pred_min': np.min(npreds),
                    })
                else:
                    pred_stats.append({'pred_mean': -1, 'pred_std': 0, 'pred_max': -1, 'pred_min': -1})
            else:
                pred_stats.append({'pred_mean': -1, 'pred_std': 0, 'pred_max': -1, 'pred_min': -1})
        
        pstat_df = pd.DataFrame(pred_stats, index=df.index)
        features['neighbor_pred_mean'] = pstat_df['pred_mean']
        features['neighbor_pred_std'] = pstat_df['pred_std']
        features['neighbor_pred_max'] = pstat_df['pred_max']
        features['neighbor_pred_min'] = pstat_df['pred_min']
    
    return features


def train_lgb(X_train, y_train, X_val, y_val, params=None, seed=SEED):
    """Train LightGBM model."""
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.02,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.6,
            'bagging_freq': 5,
            'min_data_in_leaf': 30,
            'lambda_l1': 0.3,
            'lambda_l2': 0.3,
            'verbose': -1,
            'seed': seed,
            'n_jobs': -1,
        }
    
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(150, verbose=False)]
    )
    
    return model


def train_xgb(X_train, y_train, X_val, y_val, params=None, seed=SEED):
    """Train XGBoost model."""
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'max_depth': 8,
            'learning_rate': 0.02,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'min_child_weight': 8,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'seed': seed,
            'n_jobs': -1,
        }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, 'val')],
        early_stopping_rounds=150,
        verbose_eval=False
    )
    
    return model


def train_cat(X_train, y_train, X_val, y_val, seed=SEED):
    """Train CatBoost model."""
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3,
        random_seed=seed,
        verbose=False,
        early_stopping_rounds=150,
        task_type='CPU',
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model


def predict_lgb(model, X):
    return model.predict(X)

def predict_xgb(model, X):
    return model.predict(xgb.DMatrix(X))

def predict_cat(model, X):
    return model.predict_proba(X)[:, 1]


def rank_average(predictions: List[np.ndarray]) -> np.ndarray:
    """Compute rank-averaged predictions."""
    n = len(predictions[0])
    ranks = [rankdata(p) / n for p in predictions]
    return np.mean(ranks, axis=0)


def run_cv(
    labeled_df: pd.DataFrame,
    neighbors: Dict,
    degree: Dict,
    exp_name: str = "baseline",
    use_pseudo: bool = False,
    unlabeled_df: pd.DataFrame = None,
    pseudo_weight: float = 0.1,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run cross-validation with experiment logging.
    """
    print("\n" + "=" * 60)
    print(f"CROSS-VALIDATION: {exp_name}")
    print("=" * 60)
    
    y = labeled_df['is_cheating'].values
    user_hashes = labeled_df['user_hash'].values
    n_samples = len(labeled_df)
    
    # Create tabular features
    tab_features = create_tabular_features(labeled_df)
    print(f"Tabular features: {tab_features.shape[1]}")
    
    # Initialize OOF
    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cat = np.zeros(n_samples)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(labeled_df, y)):
        print(f"\nFold {fold_idx + 1}/{N_FOLDS}")
        print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,}")
        
        # Get train fold labels for fold-safe features
        train_users = user_hashes[train_idx]
        train_labels_dict = dict(zip(train_users, y[train_idx]))
        
        # Create graph features (fold-safe)
        graph_features = create_graph_features(
            labeled_df, neighbors, degree, train_labels_dict, None
        )
        
        # Combine features
        X = pd.concat([tab_features, graph_features], axis=1)
        print(f"  Total features: {X.shape[1]}")
        
        X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Optional: add pseudo-labeled samples
        if use_pseudo and unlabeled_df is not None:
            print(f"  Adding {len(unlabeled_df)} pseudo-negatives with weight {pseudo_weight}")
            pseudo_tab = create_tabular_features(unlabeled_df)
            pseudo_graph = create_graph_features(unlabeled_df, neighbors, degree, train_labels_dict, None)
            X_pseudo = pd.concat([pseudo_tab, pseudo_graph], axis=1).values
            y_pseudo = np.zeros(len(unlabeled_df))
            
            # Sample weights for pseudo labels
            train_weights = np.ones(len(y_train))
            pseudo_weights = np.ones(len(y_pseudo)) * pseudo_weight
            
            X_train = np.vstack([X_train, X_pseudo])
            y_train = np.concatenate([y_train, y_pseudo])
            sample_weights = np.concatenate([train_weights, pseudo_weights])
        else:
            sample_weights = None
        
        # Train models
        print("  Training LGB...")
        lgb_model = train_lgb(X_train, y_train, X_val, y_val)
        oof_lgb[val_idx] = predict_lgb(lgb_model, X_val)
        
        print("  Training XGB...")
        xgb_model = train_xgb(X_train, y_train, X_val, y_val)
        oof_xgb[val_idx] = predict_xgb(xgb_model, X_val)
        
        print("  Training CAT...")
        cat_model = train_cat(X_train, y_train, X_val, y_val)
        oof_cat[val_idx] = predict_cat(cat_model, X_val)
        
        # Evaluate fold
        for name, preds in [('LGB', oof_lgb[val_idx]), ('XGB', oof_xgb[val_idx]), ('CAT', oof_cat[val_idx])]:
            result = metric.evaluate(y_val, preds, n_thresholds=N_THRESHOLDS)
            print(f"    {name}: cost={result.best_cost:,}")
        
        # Blends
        blend3 = (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx]) / 3
        rank3 = rank_average([oof_lgb[val_idx], oof_xgb[val_idx], oof_cat[val_idx]])
        
        r_blend3 = metric.evaluate(y_val, blend3, n_thresholds=N_THRESHOLDS)
        r_rank3 = metric.evaluate(y_val, rank3, n_thresholds=N_THRESHOLDS)
        
        print(f"    BLEND3: cost={r_blend3.best_cost:,}")
        print(f"    RANK3:  cost={r_rank3.best_cost:,}")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'lgb': int(metric.evaluate(y_val, oof_lgb[val_idx], n_thresholds=N_THRESHOLDS).best_cost),
            'xgb': int(metric.evaluate(y_val, oof_xgb[val_idx], n_thresholds=N_THRESHOLDS).best_cost),
            'cat': int(metric.evaluate(y_val, oof_cat[val_idx], n_thresholds=N_THRESHOLDS).best_cost),
            'blend3': int(r_blend3.best_cost),
            'rank3': int(r_rank3.best_cost),
        })
    
    # Create OOF DataFrame
    blend3_full = (oof_lgb + oof_xgb + oof_cat) / 3
    rank3_full = rank_average([oof_lgb, oof_xgb, oof_cat])
    
    oof_df = pd.DataFrame({
        'user_hash': user_hashes,
        'is_cheating': y,
        'pred_lgb': oof_lgb,
        'pred_xgb': oof_xgb,
        'pred_cat': oof_cat,
        'pred_blend3': blend3_full,
        'pred_rank3': rank3_full,
    })
    
    # Final OOF evaluation
    print("\n" + "=" * 60)
    print("OOF RESULTS")
    print("=" * 60)
    
    cv_results = {}
    for name, col in [('LGB', 'pred_lgb'), ('XGB', 'pred_xgb'), ('CAT', 'pred_cat'),
                      ('BLEND3', 'pred_blend3'), ('RANK3', 'pred_rank3')]:
        result = metric.evaluate(y, oof_df[col].values, n_thresholds=N_THRESHOLDS)
        cv_results[name] = {
            'cost': result.best_cost,
            'score': result.best_score,
            't_low': result.t_low,
            't_high': result.t_high,
        }
        print(f"{name}: cost={result.best_cost:,} | t_low={result.t_low:.3f} | t_high={result.t_high:.3f}")
    
    best_model = min(cv_results.keys(), key=lambda k: cv_results[k]['cost'])
    
    cv_summary = {
        'exp_name': exp_name,
        'n_folds': N_FOLDS,
        'seed': SEED,
        'n_samples': n_samples,
        'fold_results': fold_results,
        'cv_results': cv_results,
        'best_model': best_model,
        'timestamp': datetime.now().isoformat(),
    }
    
    return oof_df, cv_summary


def train_final_and_predict(
    labeled_df: pd.DataFrame,
    test_df: pd.DataFrame,
    neighbors: Dict,
    degree: Dict,
    oof_preds: Dict[str, float] = None,
) -> Dict[str, np.ndarray]:
    """Train on full data and predict test set."""
    print("\n" + "=" * 60)
    print("FINAL TRAINING")
    print("=" * 60)
    
    y = labeled_df['is_cheating'].values
    all_labels = dict(zip(labeled_df['user_hash'], y))
    
    # Create features
    tab_train = create_tabular_features(labeled_df)
    graph_train = create_graph_features(labeled_df, neighbors, degree, all_labels, oof_preds)
    X_train = pd.concat([tab_train, graph_train], axis=1)
    
    tab_test = create_tabular_features(test_df)
    graph_test = create_graph_features(test_df, neighbors, degree, all_labels, oof_preds)
    X_test = pd.concat([tab_test, graph_test], axis=1)
    
    # Ensure same columns
    X_test = X_test[X_train.columns]
    
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    
    # Train models
    print("\nTraining LGB...")
    lgb_model = train_lgb(X_train.values, y, X_train.values, y)
    
    print("Training XGB...")
    xgb_model = train_xgb(X_train.values, y, X_train.values, y)
    
    print("Training CAT...")
    cat_model = train_cat(X_train.values, y, X_train.values, y)
    
    # Predict
    pred_lgb = predict_lgb(lgb_model, X_test.values)
    pred_xgb = predict_xgb(xgb_model, X_test.values)
    pred_cat = predict_cat(cat_model, X_test.values)
    
    pred_blend3 = (pred_lgb + pred_xgb + pred_cat) / 3
    pred_rank3 = rank_average([pred_lgb, pred_xgb, pred_cat])
    
    print(f"\nPrediction ranges:")
    for name, pred in [('LGB', pred_lgb), ('XGB', pred_xgb), ('CAT', pred_cat),
                       ('BLEND3', pred_blend3), ('RANK3', pred_rank3)]:
        print(f"  {name}: [{pred.min():.4f}, {pred.max():.4f}]")
    
    return {
        'lgb': pred_lgb,
        'xgb': pred_xgb,
        'cat': pred_cat,
        'blend3': pred_blend3,
        'rank3': pred_rank3,
    }


def create_submission(test_df: pd.DataFrame, predictions: np.ndarray, 
                     sample_sub: pd.DataFrame, output_path: str):
    """Create submission file."""
    submission = pd.DataFrame({
        'user_hash': test_df['user_hash'],
        'prediction': predictions
    })
    submission = sample_sub[['user_hash']].merge(submission, on='user_hash', how='left')
    
    assert len(submission) == len(sample_sub)
    assert submission['prediction'].notna().all()
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path} ({len(submission):,} rows)")
    return submission


def main():
    print("=" * 60)
    print("MERCOR CHEATING DETECTION - IMPROVED PIPELINE v2")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load data
    train, test, graph, sample_sub = load_data(".")
    labeled, unlabeled = split_data(train)
    
    # Build graph
    print("\nBuilding graph...")
    all_users = set(train['user_hash'].tolist() + test['user_hash'].tolist())
    neighbors, degree = build_graph(graph, all_users)
    print(f"  Nodes: {len(neighbors):,} | Max degree: {max(degree.values())}")
    
    # Create experiment tracker
    tracker = ExperimentTracker(labeled)
    
    # Experiment 1: Baseline with improved features
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Improved baseline")
    print("=" * 70)
    
    oof_df, cv_summary = run_cv(labeled, neighbors, degree, exp_name="improved_baseline_v2")
    
    # Evaluate on proxy splits
    for pred_col in ['pred_blend3', 'pred_rank3']:
        results = tracker.evaluate_oof(oof_df.rename(columns={pred_col: 'prediction'}), 'prediction')
        print(f"\n{pred_col}:")
        print(f"  CV: {results['cv_cost']:,} | Public: {results['public_cost']:,} | Private: {results['private_cost']:,}")
    
    # Select best blend column
    best_blend = 'pred_blend3'
    best_cv_cost = cv_summary['cv_results']['BLEND3']['cost']
    if cv_summary['cv_results']['RANK3']['cost'] < best_cv_cost:
        best_blend = 'pred_rank3'
        best_cv_cost = cv_summary['cv_results']['RANK3']['cost']
    
    # Log experiment
    oof_for_log = oof_df.rename(columns={best_blend: 'prediction'})
    exp_id = tracker.log_and_save(
        exp_name="improved_baseline_v2",
        oof_df=oof_for_log,
        cv_summary=cv_summary,
        pred_col='prediction',
        notes=f"Best blend: {best_blend}"
    )
    
    # Train final models and create submission
    oof_preds = dict(zip(oof_df['user_hash'], oof_df[best_blend]))
    test_preds = train_final_and_predict(labeled, test, neighbors, degree, oof_preds)
    
    # Create submissions
    create_submission(test, test_preds['blend3'], sample_sub, "submission.csv")
    create_submission(test, test_preds['rank3'], sample_sub, "submission_rank.csv")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    best_result = cv_summary['cv_results'][cv_summary['best_model']]
    print(f"Best model: {cv_summary['best_model']}")
    print(f"CV Cost: {best_result['cost']:,}")
    print(f"CV Score: {best_result['score']:,}")
    print(f"Thresholds: t_low={best_result['t_low']:.4f}, t_high={best_result['t_high']:.4f}")
    
    return cv_summary


if __name__ == "__main__":
    cv_summary = main()
