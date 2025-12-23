#!/usr/bin/env python3
"""
MAXIMUM-POWER Competition Pipeline for Mercor Cheating Detection.

This script implements:
1. Advanced tabular features with missingness exploitation
2. Deep graph features (degree, centrality, components, embeddings)
3. Multiple PU learning schemes
4. Ensemble of CatBoost, LightGBM, XGBoost with tuned hyperparameters
5. Graph score propagation
6. Meta-model stacking
7. Calibration and post-processing
"""
import os
import sys
import json
import pickle
import hashlib
import time
import warnings
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# Configuration
SEED = 42
np.random.seed(SEED)

DATA_DIR = "/home/runner/work/Mercor-Cheating-Detection/Mercor-Cheating-Detection"
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")

# Cost metric parameters
COST_FN = 500   # False negative: auto-pass a cheater
COST_REVIEW = 25  # Manual review
COST_FP = 100   # False positive: auto-block clean user

# Feature columns
FEATURE_COLS = [f"feature_{i:03d}" for i in range(1, 19)]


def load_data():
    """Load all data files."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    graph = pd.read_csv(os.path.join(DATA_DIR, "social_graph.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    
    print(f"Train: {len(train):,} rows")
    print(f"Test: {len(test):,} rows")
    print(f"Graph edges: {len(graph):,}")
    
    # Data summary
    labeled_mask = train['is_cheating'].notna()
    print(f"\nLabeled: {labeled_mask.sum():,}")
    print(f"Unlabeled (high_conf_clean): {(~labeled_mask).sum():,}")
    print(f"Cheating rate (labeled): {train.loc[labeled_mask, 'is_cheating'].mean():.2%}")
    
    return train, test, graph, sample_sub


def compute_cost(y_true, y_pred, low_thresh, high_thresh):
    """Compute total cost for given thresholds."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    auto_pass = y_pred < low_thresh
    manual_review = (y_pred >= low_thresh) & (y_pred < high_thresh)
    auto_block = y_pred >= high_thresh
    
    cost = 0.0
    cost += COST_FN * (auto_pass & (y_true == 1)).sum()
    cost += COST_REVIEW * manual_review.sum()
    cost += COST_FP * (auto_block & (y_true == 0)).sum()
    
    return cost


def find_optimal_thresholds(y_true, y_pred, n_search=100):
    """Find optimal thresholds to minimize cost."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Use quantiles for threshold candidates
    thresholds = np.percentile(y_pred, np.linspace(0, 100, n_search))
    thresholds = np.unique(np.concatenate([[0.0], thresholds, [1.0]]))
    
    best_cost = float('inf')
    best_low = 0.0
    best_high = 1.0
    
    for low_thresh in thresholds:
        for high_thresh in thresholds:
            if high_thresh <= low_thresh:
                continue
            cost = compute_cost(y_true, y_pred, low_thresh, high_thresh)
            if cost < best_cost:
                best_cost = cost
                best_low = low_thresh
                best_high = high_thresh
    
    return best_low, best_high, best_cost


def evaluate(y_true, y_pred):
    """Evaluate predictions and return metrics."""
    low, high, cost = find_optimal_thresholds(y_true, y_pred)
    return {
        'cost': cost,
        'score': -cost,  # Kaggle format (higher is better)
        'low_thresh': low,
        'high_thresh': high
    }


class UnionFind:
    """Union-Find for connected components."""
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def compute_graph_features(graph_df, cache_path=None):
    """Compute comprehensive graph features."""
    print("\n" + "=" * 60)
    print("COMPUTING GRAPH FEATURES")
    print("=" * 60)
    
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached graph features from {cache_path}")
        return pd.read_pickle(cache_path)
    
    # Build adjacency list
    print("Building adjacency list...")
    adj = defaultdict(set)
    for _, row in tqdm(graph_df.iterrows(), total=len(graph_df), desc="Adjacency"):
        adj[row['user_a']].add(row['user_b'])
        adj[row['user_b']].add(row['user_a'])
    
    all_nodes = list(adj.keys())
    print(f"Total nodes in graph: {len(all_nodes):,}")
    
    # Compute connected components
    print("Computing connected components...")
    uf = UnionFind()
    for _, row in tqdm(graph_df.iterrows(), total=len(graph_df), desc="Components"):
        uf.union(row['user_a'], row['user_b'])
    
    component_map = {}
    root_to_id = {}
    comp_id = 0
    for node in all_nodes:
        root = uf.find(node)
        if root not in root_to_id:
            root_to_id[root] = comp_id
            comp_id += 1
        component_map[node] = root_to_id[root]
    
    # Compute component sizes
    comp_sizes = defaultdict(int)
    for node, cid in component_map.items():
        comp_sizes[cid] += 1
    
    print(f"Found {comp_id:,} connected components")
    
    # Compute node features
    print("Computing node features...")
    features = []
    
    for node in tqdm(all_nodes, desc="Node features"):
        neighbors = adj[node]
        degree = len(neighbors)
        
        # Basic degree features
        feat = {
            'user_hash': node,
            'degree': degree,
            'log_degree': np.log1p(degree),
            'component_id': component_map[node],
            'component_size': comp_sizes[component_map[node]],
            'log_component_size': np.log1p(comp_sizes[component_map[node]]),
        }
        
        # Second-order features (neighbor statistics)
        if degree > 0:
            neighbor_degrees = [len(adj[n]) for n in neighbors]
            feat['neighbor_degree_mean'] = np.mean(neighbor_degrees)
            feat['neighbor_degree_std'] = np.std(neighbor_degrees) if len(neighbor_degrees) > 1 else 0
            feat['neighbor_degree_max'] = np.max(neighbor_degrees)
            feat['neighbor_degree_min'] = np.min(neighbor_degrees)
            
            # Clustering coefficient approximation
            triangles = 0
            neighbor_list = list(neighbors)
            for i, n1 in enumerate(neighbor_list[:min(50, len(neighbor_list))]):  # Limit for speed
                for n2 in neighbor_list[i+1:min(50, len(neighbor_list))]:
                    if n2 in adj[n1]:
                        triangles += 1
            max_triangles = degree * (degree - 1) / 2
            feat['clustering_coef'] = triangles / max_triangles if max_triangles > 0 else 0
        else:
            feat['neighbor_degree_mean'] = 0
            feat['neighbor_degree_std'] = 0
            feat['neighbor_degree_max'] = 0
            feat['neighbor_degree_min'] = 0
            feat['clustering_coef'] = 0
        
        features.append(feat)
    
    graph_feats = pd.DataFrame(features)
    
    # Degree centrality (normalized)
    total_nodes = len(all_nodes)
    graph_feats['degree_centrality'] = graph_feats['degree'] / (total_nodes - 1)
    
    # Component-relative degree
    graph_feats['degree_in_component'] = graph_feats.apply(
        lambda x: x['degree'] / x['component_size'] if x['component_size'] > 1 else 0, axis=1
    )
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        graph_feats.to_pickle(cache_path)
        print(f"Saved graph features to {cache_path}")
    
    return graph_feats, component_map, adj


def compute_node2vec_embeddings(graph_df, dim=64, walks_per_node=10, walk_length=20, 
                                 p=1.0, q=1.0, cache_path=None):
    """Compute Node2Vec embeddings."""
    print("\n" + "=" * 60)
    print("COMPUTING NODE2VEC EMBEDDINGS")
    print("=" * 60)
    
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return pd.read_pickle(cache_path)
    
    from gensim.models import Word2Vec
    
    # Build adjacency list
    adj = defaultdict(list)
    for _, row in tqdm(graph_df.iterrows(), total=len(graph_df), desc="Building graph"):
        adj[row['user_a']].append(row['user_b'])
        adj[row['user_b']].append(row['user_a'])
    
    nodes = list(adj.keys())
    print(f"Nodes: {len(nodes):,}")
    
    # Generate random walks
    print(f"Generating {walks_per_node} walks per node, length {walk_length}...")
    walks = []
    
    for _ in tqdm(range(walks_per_node), desc="Walk iterations"):
        np.random.shuffle(nodes)
        for start_node in nodes:
            walk = [start_node]
            
            if start_node not in adj or len(adj[start_node]) == 0:
                walks.append(walk)
                continue
            
            walk.append(np.random.choice(adj[start_node]))
            
            while len(walk) < walk_length:
                cur = walk[-1]
                prev = walk[-2]
                
                if cur not in adj or len(adj[cur]) == 0:
                    break
                
                neighbors = adj[cur]
                probs = []
                
                for neighbor in neighbors:
                    if neighbor == prev:
                        probs.append(1.0 / p)
                    elif neighbor in adj.get(prev, []):
                        probs.append(1.0)
                    else:
                        probs.append(1.0 / q)
                
                probs = np.array(probs)
                probs = probs / probs.sum()
                next_node = np.random.choice(neighbors, p=probs)
                walk.append(next_node)
            
            walks.append(walk)
    
    print(f"Generated {len(walks):,} walks")
    
    # Train Word2Vec
    print("Training Word2Vec...")
    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=5,
        min_count=1,
        sg=1,
        workers=4,
        seed=SEED,
        epochs=5
    )
    
    # Extract embeddings
    emb_cols = [f'n2v_{i}' for i in range(dim)]
    data = []
    for node in tqdm(nodes, desc="Extracting embeddings"):
        if node in model.wv:
            emb = model.wv[node]
            data.append({'user_hash': node, **{col: emb[i] for i, col in enumerate(emb_cols)}})
    
    emb_df = pd.DataFrame(data)
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        emb_df.to_pickle(cache_path)
        print(f"Saved embeddings to {cache_path}")
    
    return emb_df


def create_tabular_features(df, is_train=True):
    """Create comprehensive tabular features."""
    df = df.copy()
    
    # Missing indicators
    for col in FEATURE_COLS:
        df[f'{col}_missing'] = df[col].isna().astype(int)
    
    # Fill missing values
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(-999)
    
    # Feature interactions (selected)
    df['feat_001_002_sum'] = df['feature_001'] + df['feature_002']
    df['feat_001_002_diff'] = df['feature_001'] - df['feature_002']
    df['feat_004_005_ratio'] = df['feature_004'] / (df['feature_005'] + 1)
    df['feat_008_009_sum'] = df['feature_008'] + df['feature_009']
    df['feat_015_016_ratio'] = df['feature_015'] / (df['feature_016'] + 1)
    
    # Total missing count
    missing_cols = [c for c in df.columns if c.endswith('_missing')]
    df['total_missing'] = df[missing_cols].sum(axis=1)
    
    return df


def prepare_features(train, test, graph_feats, embeddings=None):
    """Prepare all features for modeling."""
    print("\n" + "=" * 60)
    print("PREPARING FEATURES")
    print("=" * 60)
    
    # Create tabular features
    train = create_tabular_features(train, is_train=True)
    test = create_tabular_features(test, is_train=False)
    
    # Merge graph features
    train = train.merge(graph_feats, on='user_hash', how='left')
    test = test.merge(graph_feats, on='user_hash', how='left')
    
    # Merge embeddings if provided
    if embeddings is not None:
        train = train.merge(embeddings, on='user_hash', how='left')
        test = test.merge(embeddings, on='user_hash', how='left')
    
    # Fill missing graph features (users not in graph)
    graph_cols = [c for c in graph_feats.columns if c != 'user_hash']
    for col in graph_cols:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)
    
    if embeddings is not None:
        emb_cols = [c for c in embeddings.columns if c != 'user_hash']
        for col in emb_cols:
            train[col] = train[col].fillna(0)
            test[col] = test[col].fillna(0)
    
    # Get feature columns
    exclude_cols = ['user_hash', 'is_cheating', 'high_conf_clean', 'component_id']
    feature_cols = [c for c in train.columns if c not in exclude_cols]
    
    print(f"Total features: {len(feature_cols)}")
    
    return train, test, feature_cols


def component_group_split(df, component_map, n_splits=5):
    """Create CV splits based on connected components to avoid leakage."""
    df = df.copy()
    df['component_id'] = df['user_hash'].map(component_map)
    
    # For users not in graph, assign unique component IDs
    max_comp = max(component_map.values()) + 1 if component_map else 0
    missing_mask = df['component_id'].isna()
    df.loc[missing_mask, 'component_id'] = np.arange(max_comp, max_comp + missing_mask.sum())
    df['component_id'] = df['component_id'].astype(int)
    
    groups = df['component_id'].values
    gkf = GroupKFold(n_splits=n_splits)
    
    return list(gkf.split(df, groups=groups))


def train_catboost(X_train, y_train, X_val, y_val, sample_weight=None, params=None):
    """Train CatBoost with custom parameters."""
    default_params = {
        'iterations': 3000,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 3,
        'random_seed': SEED,
        'verbose': 200,
        'early_stopping_rounds': 200,
        'task_type': 'CPU',
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
    }
    if params:
        default_params.update(params)
    
    train_pool = cb.Pool(X_train, y_train, weight=sample_weight)
    val_pool = cb.Pool(X_val, y_val)
    
    model = cb.CatBoostClassifier(**default_params)
    model.fit(train_pool, eval_set=val_pool)
    
    val_preds = model.predict_proba(X_val)[:, 1]
    return model, val_preds


def train_lightgbm(X_train, y_train, X_val, y_val, sample_weight=None, params=None):
    """Train LightGBM with custom parameters."""
    default_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_child_samples': 20,
        'seed': SEED,
        'verbose': -1,
        'n_estimators': 3000,
    }
    if params:
        default_params.update(params)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=200)
    ]
    
    model = lgb.LGBMClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weight,
        callbacks=callbacks
    )
    
    val_preds = model.predict_proba(X_val)[:, 1]
    return model, val_preds


def train_xgboost(X_train, y_train, X_val, y_val, sample_weight=None, params=None):
    """Train XGBoost with custom parameters."""
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'max_depth': 8,
        'eta': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': SEED,
        'verbosity': 0,
        'n_estimators': 3000,
    }
    if params:
        default_params.update(params)
    
    model = xgb.XGBClassifier(**default_params, early_stopping_rounds=200)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weight,
        verbose=200
    )
    
    val_preds = model.predict_proba(X_val)[:, 1]
    return model, val_preds


def create_pu_weights(y, high_conf_clean_mask, pu_weight=0.05, class_balance=True):
    """Create sample weights for PU learning."""
    weights = np.ones(len(y), dtype=float)
    
    # Class balancing for labeled data
    if class_balance:
        labeled_mask = (y == 0) | (y == 1)
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        
        if n_pos > 0 and n_neg > 0:
            pos_weight = len(y[labeled_mask]) / (2 * n_pos)
            neg_weight = len(y[labeled_mask]) / (2 * n_neg)
            weights[y == 1] = pos_weight
            weights[y == 0] = neg_weight
    
    # PU weight for high_conf_clean
    weights[high_conf_clean_mask] = pu_weight
    
    return weights


def run_cv(df, feature_cols, component_map, model_type='catboost', 
           n_folds=5, pu_weight=0.05, params=None):
    """Run cross-validation."""
    print(f"\n--- Training {model_type.upper()} ({n_folds} folds) ---")
    
    # Get labeled data
    labeled_mask = df['is_cheating'].notna()
    labeled_df = df[labeled_mask].copy()
    
    X = labeled_df[feature_cols].values
    y = labeled_df['is_cheating'].values.astype(float)
    high_conf_clean = labeled_df['high_conf_clean'].values
    
    # Create CV splits
    splits = component_group_split(labeled_df, component_map, n_folds)
    
    oof_preds = np.full(len(labeled_df), np.nan)
    models = []
    
    trainers = {
        'catboost': train_catboost,
        'lightgbm': train_lightgbm,
        'xgboost': train_xgboost,
    }
    trainer = trainers[model_type]
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create PU weights
        train_hcc = high_conf_clean[train_idx]
        sample_weight = create_pu_weights(
            y_train,
            high_conf_clean_mask=(train_hcc == 1),
            pu_weight=pu_weight
        )
        
        model, val_preds = trainer(X_train, y_train, X_val, y_val, sample_weight, params)
        oof_preds[val_idx] = val_preds
        models.append(model)
        
        # Fold evaluation
        result = evaluate(y_val, val_preds)
        print(f"Fold {fold + 1} cost: {result['cost']:,.0f}")
    
    # Overall evaluation
    result = evaluate(y, oof_preds)
    print(f"\n{model_type.upper()} OOF Cost: {result['cost']:,.0f} (Score: {result['score']:,.0f})")
    
    return oof_preds, models, labeled_df.index


def graph_propagation(preds_df, adj, alpha=0.2, n_iter=3):
    """Propagate scores through the graph."""
    score_map = dict(zip(preds_df['user_hash'], preds_df['prediction']))
    
    for _ in range(n_iter):
        new_scores = {}
        for user, score in score_map.items():
            if user in adj and len(adj[user]) > 0:
                neighbors = adj[user]
                neighbor_scores = [score_map.get(n, 0.5) for n in neighbors]
                neighbor_mean = np.mean(neighbor_scores)
                new_scores[user] = (1 - alpha) * score + alpha * neighbor_mean
            else:
                new_scores[user] = score
        score_map = new_scores
    
    return np.array([score_map.get(u, preds_df.loc[preds_df['user_hash'] == u, 'prediction'].values[0])
                     for u in preds_df['user_hash']])


def blend_predictions(pred_dict, weights=None, method='linear'):
    """Blend predictions from multiple models."""
    if weights is None:
        weights = {k: 1.0 / len(pred_dict) for k in pred_dict}
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    if method == 'linear':
        result = np.zeros_like(list(pred_dict.values())[0])
        for name, preds in pred_dict.items():
            result += weights.get(name, 0) * preds
        return result
    
    elif method == 'rank':
        n = len(list(pred_dict.values())[0])
        rank_sum = np.zeros(n)
        for name, preds in pred_dict.items():
            ranks = rankdata(preds) / n
            rank_sum += weights.get(name, 0) * ranks
        return rank_sum
    
    elif method == 'logit':
        eps = 1e-7
        logit_sum = np.zeros_like(list(pred_dict.values())[0])
        for name, preds in pred_dict.items():
            preds_clipped = np.clip(preds, eps, 1 - eps)
            logits = np.log(preds_clipped / (1 - preds_clipped))
            logit_sum += weights.get(name, 0) * logits
        return 1 / (1 + np.exp(-logit_sum))


def optimize_blend_weights(pred_dict, y_true, method='linear'):
    """Optimize blend weights using grid search."""
    names = list(pred_dict.keys())
    n_models = len(names)
    
    best_cost = float('inf')
    best_weights = {n: 1.0 / n_models for n in names}
    
    if n_models == 3:
        for w1 in np.linspace(0.1, 0.8, 8):
            for w2 in np.linspace(0.1, 0.9 - w1, 8):
                w3 = 1 - w1 - w2
                if w3 < 0.05:
                    continue
                
                weights = {names[0]: w1, names[1]: w2, names[2]: w3}
                blended = blend_predictions(pred_dict, weights, method)
                _, _, cost = find_optimal_thresholds(y_true, blended)
                
                if cost < best_cost:
                    best_cost = cost
                    best_weights = weights.copy()
    
    return best_weights, best_cost


def main():
    start_time = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ARTIFACTS_DIR, f"run_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"MAXIMUM-POWER PIPELINE - Run {run_id}")
    print("=" * 60)
    
    # Load data
    train, test, graph, sample_sub = load_data()
    
    # Compute graph features
    graph_cache = os.path.join(ARTIFACTS_DIR, "graph_features.pkl")
    result = compute_graph_features(graph, cache_path=None)
    if isinstance(result, tuple):
        graph_feats, component_map, adj = result
    else:
        graph_feats = result
        component_map = {}
        adj = {}
    
    # Compute Node2Vec embeddings
    emb_cache = os.path.join(ARTIFACTS_DIR, "node2vec_embeddings.pkl")
    embeddings = compute_node2vec_embeddings(graph, dim=32, walks_per_node=5, walk_length=15,
                                              cache_path=emb_cache)
    
    # Prepare features
    train, test, feature_cols = prepare_features(train, test, graph_feats, embeddings)
    
    # Get labeled data info
    labeled_mask = train['is_cheating'].notna()
    labeled_df = train[labeled_mask].copy()
    y_true = labeled_df['is_cheating'].values.astype(int)
    
    # Run CV for each model type
    oof_results = {}
    all_models = {}
    
    for model_type in ['catboost', 'lightgbm', 'xgboost']:
        oof_preds, models, idx = run_cv(
            train, feature_cols, component_map,
            model_type=model_type,
            n_folds=5,
            pu_weight=0.05
        )
        oof_results[model_type] = oof_preds
        all_models[model_type] = models
    
    # Find best blend
    print("\n" + "=" * 60)
    print("OPTIMIZING ENSEMBLE")
    print("=" * 60)
    
    best_blend = None
    best_blend_cost = float('inf')
    best_blend_method = None
    best_weights = None
    
    for method in ['linear', 'rank', 'logit']:
        weights, cost = optimize_blend_weights(oof_results, y_true, method)
        print(f"{method}: cost={cost:,.0f}, weights={weights}")
        
        if cost < best_blend_cost:
            best_blend_cost = cost
            best_blend_method = method
            best_weights = weights
            best_blend = blend_predictions(oof_results, weights, method)
    
    print(f"\nBest blend: {best_blend_method}, cost={best_blend_cost:,.0f}")
    
    # Try propagation
    print("\n" + "=" * 60)
    print("TESTING GRAPH PROPAGATION")
    print("=" * 60)
    
    best_prop_cost = best_blend_cost
    best_prop_preds = best_blend
    best_prop_config = None
    
    for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for n_iter in [2, 3, 5]:
            prop_df = pd.DataFrame({
                'user_hash': labeled_df['user_hash'].values,
                'prediction': best_blend
            })
            prop_preds = graph_propagation(prop_df, adj, alpha=alpha, n_iter=n_iter)
            result = evaluate(y_true, prop_preds)
            
            if result['cost'] < best_prop_cost:
                best_prop_cost = result['cost']
                best_prop_preds = prop_preds
                best_prop_config = {'alpha': alpha, 'n_iter': n_iter}
                print(f"alpha={alpha}, n_iter={n_iter}: cost={result['cost']:,.0f} [NEW BEST]")
            else:
                print(f"alpha={alpha}, n_iter={n_iter}: cost={result['cost']:,.0f}")
    
    if best_prop_config:
        print(f"\nBest propagation: {best_prop_config}, cost={best_prop_cost:,.0f}")
        final_oof = best_prop_preds
    else:
        final_oof = best_blend
        best_prop_config = None
    
    # Final evaluation
    final_result = evaluate(y_true, final_oof)
    print("\n" + "=" * 60)
    print("FINAL OOF RESULTS")
    print("=" * 60)
    print(f"Cost: {final_result['cost']:,.0f}")
    print(f"Score: {final_result['score']:,.0f}")
    print(f"Thresholds: low={final_result['low_thresh']:.4f}, high={final_result['high_thresh']:.4f}")
    
    # Generate test predictions
    print("\n" + "=" * 60)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 60)
    
    X_test = test[feature_cols].values
    test_preds = {}
    
    for model_type, models in all_models.items():
        preds = np.zeros(len(test))
        for model in models:
            preds += model.predict_proba(X_test)[:, 1] / len(models)
        test_preds[model_type] = preds
    
    # Blend test predictions
    final_test_preds = blend_predictions(test_preds, best_weights, best_blend_method)
    
    # Apply propagation if it helped
    if best_prop_config:
        test_prop_df = pd.DataFrame({
            'user_hash': test['user_hash'].values,
            'prediction': final_test_preds
        })
        final_test_preds = graph_propagation(test_prop_df, adj, 
                                              alpha=best_prop_config['alpha'],
                                              n_iter=best_prop_config['n_iter'])
    
    # Create submission
    submission = pd.DataFrame({
        'user_hash': test['user_hash'],
        'prediction': final_test_preds
    })
    
    # Ensure correct order
    submission = submission.set_index('user_hash').loc[sample_sub['user_hash']].reset_index()
    
    # Save submission
    sub_path = os.path.join(out_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    
    # Checksum
    with open(sub_path, 'rb') as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    
    # Save config
    config = {
        'run_id': run_id,
        'cv_cost': final_result['cost'],
        'cv_score': final_result['score'],
        'blend_method': best_blend_method,
        'blend_weights': best_weights,
        'propagation': best_prop_config,
        'n_features': len(feature_cols),
        'checksum': checksum,
    }
    
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Save models
    for model_type, models in all_models.items():
        for i, model in enumerate(models):
            model_path = os.path.join(out_dir, f"{model_type}_fold{i}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"CV Score: {final_result['score']:,.0f}")
    print(f"Submission: {sub_path}")
    print(f"Checksum: {checksum}")
    print("=" * 60)
    
    # Update scoreboard
    scoreboard_path = os.path.join(ARTIFACTS_DIR, "scoreboard.csv")
    entry = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'cv_cost': final_result['cost'],
        'cv_score': final_result['score'],
        'public_lb_score': None,
        'target_lb_score': -1540000,
        'beat_top1': False,
        'notes': f'{best_blend_method} blend, prop={best_prop_config}'
    }
    
    if os.path.exists(scoreboard_path):
        scoreboard = pd.read_csv(scoreboard_path)
        scoreboard = pd.concat([scoreboard, pd.DataFrame([entry])], ignore_index=True)
    else:
        scoreboard = pd.DataFrame([entry])
    scoreboard.to_csv(scoreboard_path, index=False)
    
    return config


if __name__ == "__main__":
    main()
