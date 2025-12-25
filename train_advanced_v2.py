"""
Mercor Cheating Detection - Aggressive Optimization

More experiments to push toward top 1:
1. More aggressive feature engineering
2. Pseudo-labeling with high_conf_clean rows
3. Graph propagation features
4. Hyperparameter search
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata

import metric

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
N_THRESHOLDS = 300
ARTIFACTS_DIR = "artifacts"


def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    graph = pd.read_csv("social_graph.csv")
    sample_sub = pd.read_csv("sample_submission.csv")
    return train, test, graph, sample_sub


def split_data(train):
    labeled = train[train['is_cheating'].notna()].copy()
    unlabeled = train[train['is_cheating'].isna()].copy()
    labeled['is_cheating'] = labeled['is_cheating'].astype(int)
    return labeled, unlabeled


def build_graph(graph_df, all_users):
    neighbors = defaultdict(list)
    for _, row in graph_df.iterrows():
        u, v = row['user_a'], row['user_b']
        neighbors[u].append(v)
        neighbors[v].append(u)
    for u in all_users:
        if u not in neighbors:
            neighbors[u] = []
    degree = {u: len(n) for u, n in neighbors.items()}
    return dict(neighbors), degree


def get_hop2_neighbors(user, neighbors):
    """Get 2-hop neighbors."""
    hop1 = set(neighbors.get(user, []))
    hop2 = set()
    for n in hop1:
        hop2.update(neighbors.get(n, []))
    hop2 -= hop1
    hop2.discard(user)
    return hop2


def create_advanced_features(df, neighbors, degree, train_labels=None, train_preds=None):
    """Create advanced features with more engineering."""
    feature_cols = [f"feature_{i:03d}" for i in range(1, 19)]
    
    feats = df[feature_cols].copy()
    
    # Missingness
    for col in feature_cols:
        feats[f"{col}_miss"] = feats[col].isna().astype(int)
    
    # Row statistics
    feats['na_count'] = df[feature_cols].isna().sum(axis=1)
    feats['row_mean'] = df[feature_cols].mean(axis=1)
    feats['row_std'] = df[feature_cols].std(axis=1)
    feats['row_min'] = df[feature_cols].min(axis=1)
    feats['row_max'] = df[feature_cols].max(axis=1)
    feats['row_range'] = feats['row_max'] - feats['row_min']
    feats['row_sum'] = df[feature_cols].sum(axis=1)
    feats['row_median'] = df[feature_cols].median(axis=1)
    
    # Advanced row stats
    feats['row_q25'] = df[feature_cols].quantile(0.25, axis=1)
    feats['row_q75'] = df[feature_cols].quantile(0.75, axis=1)
    feats['row_iqr'] = feats['row_q75'] - feats['row_q25']
    
    # Log transforms for wide-range features
    for col in ['feature_010', 'feature_015', 'feature_016']:
        val = df[col].fillna(0)
        feats[f'{col}_log'] = np.log1p(np.abs(val)) * np.sign(val)
        feats[f'{col}_sqrt'] = np.sqrt(np.abs(val)) * np.sign(val)
    
    # Binning
    f10 = df['feature_010'].fillna(0)
    feats['f010_bin'] = pd.cut(f10, bins=[-np.inf, 0, 100, 500, 1000, 5000, np.inf],
                               labels=[0, 1, 2, 3, 4, 5]).astype(float)
    feats['f010_nonzero'] = (f10 > 0).astype(int)
    feats['f010_high'] = (f10 > 1000).astype(int)
    
    f15 = df['feature_015'].fillna(0)
    feats['f015_bin'] = pd.cut(f15, bins=[-np.inf, 0, 1, 10, 50, 100, np.inf],
                               labels=[0, 1, 2, 3, 4, 5]).astype(float)
    feats['f015_high'] = (f15 > 50).astype(int)
    
    # Ratios
    feats['ratio_018_017'] = df['feature_018'].fillna(0) / (df['feature_017'].fillna(0) + 0.01)
    feats['ratio_f10_f15'] = (f10 + 1) / (np.abs(f15) + 1)
    feats['f016_x_f017'] = df['feature_016'].fillna(0) * df['feature_017'].fillna(0)
    
    # Binary feature interactions
    for col in ['feature_007', 'feature_011', 'feature_013', 'feature_014']:
        val = df[col].fillna(0)
        feats[f'{col}_x_na'] = val * feats['na_count']
    
    # Graph features
    feats['degree'] = df['user_hash'].map(degree).fillna(0)
    feats['log_degree'] = np.log1p(feats['degree'])
    feats['sqrt_degree'] = np.sqrt(feats['degree'])
    feats['is_isolated'] = (feats['degree'] == 0).astype(int)
    feats['high_degree'] = (feats['degree'] > 100).astype(int)
    
    # Neighbor statistics
    n_stats = []
    for user in df['user_hash']:
        ns = neighbors.get(user, [])
        if ns:
            ndeg = [degree.get(n, 0) for n in ns]
            hop2 = get_hop2_neighbors(user, neighbors)
            hop2_deg = [degree.get(n, 0) for n in hop2] if hop2 else [0]
            
            n_stats.append({
                'n_mean': np.mean(ndeg),
                'n_std': np.std(ndeg) if len(ndeg) > 1 else 0,
                'n_min': np.min(ndeg),
                'n_max': np.max(ndeg),
                'n_sum': np.sum(ndeg),
                'n_count': len(ns),
                'hop2_count': len(hop2),
                'hop2_mean': np.mean(hop2_deg),
            })
        else:
            n_stats.append({
                'n_mean': 0, 'n_std': 0, 'n_min': 0, 'n_max': 0,
                'n_sum': 0, 'n_count': 0, 'hop2_count': 0, 'hop2_mean': 0,
            })
    
    ndf = pd.DataFrame(n_stats, index=df.index)
    for col in ndf.columns:
        feats[col] = ndf[col]
    
    # Fold-safe label features
    if train_labels:
        label_feats = []
        for user in df['user_hash']:
            ns = neighbors.get(user, [])
            if ns:
                labeled_ns = [train_labels.get(n) for n in ns if n in train_labels]
                if labeled_ns:
                    hop2 = get_hop2_neighbors(user, neighbors)
                    labeled_hop2 = [train_labels.get(n) for n in hop2 if n in train_labels]
                    
                    label_feats.append({
                        'n_labeled': len(labeled_ns),
                        'cheat_rate': np.mean(labeled_ns),
                        'cheat_sum': np.sum(labeled_ns),
                        'n_labeled_hop2': len(labeled_hop2) if labeled_hop2 else 0,
                        'cheat_rate_hop2': np.mean(labeled_hop2) if labeled_hop2 else -1,
                    })
                else:
                    label_feats.append({
                        'n_labeled': 0, 'cheat_rate': -1, 'cheat_sum': 0,
                        'n_labeled_hop2': 0, 'cheat_rate_hop2': -1,
                    })
            else:
                label_feats.append({
                    'n_labeled': 0, 'cheat_rate': -1, 'cheat_sum': 0,
                    'n_labeled_hop2': 0, 'cheat_rate_hop2': -1,
                })
        
        ldf = pd.DataFrame(label_feats, index=df.index)
        feats['n_labeled'] = ldf['n_labeled']
        feats['cheat_rate'] = ldf['cheat_rate']
        feats['cheat_sum'] = ldf['cheat_sum']
        feats['labeled_ratio'] = ldf['n_labeled'] / (feats['degree'] + 1)
        feats['n_labeled_hop2'] = ldf['n_labeled_hop2']
        feats['cheat_rate_hop2'] = ldf['cheat_rate_hop2']
    
    # Prediction propagation
    if train_preds:
        pred_feats = []
        for user in df['user_hash']:
            ns = neighbors.get(user, [])
            if ns:
                npreds = [train_preds.get(n) for n in ns if n in train_preds]
                if npreds:
                    pred_feats.append({
                        'pred_mean': np.mean(npreds),
                        'pred_std': np.std(npreds) if len(npreds) > 1 else 0,
                        'pred_max': np.max(npreds),
                        'pred_min': np.min(npreds),
                    })
                else:
                    pred_feats.append({'pred_mean': -1, 'pred_std': 0, 'pred_max': -1, 'pred_min': -1})
            else:
                pred_feats.append({'pred_mean': -1, 'pred_std': 0, 'pred_max': -1, 'pred_min': -1})
        
        pdf = pd.DataFrame(pred_feats, index=df.index)
        for col in pdf.columns:
            feats[col] = pdf[col]
    
    feats = feats.fillna(-999)
    return feats


def train_lgb_v2(X_train, y_train, X_val, y_val, seed=SEED):
    """LightGBM with more aggressive params."""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 127,
        'learning_rate': 0.03,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_data_in_leaf': 30,
        'lambda_l1': 0.2,
        'lambda_l2': 0.2,
        'verbose': -1,
        'seed': seed,
        'n_jobs': -1,
    }
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    model = lgb.train(params, train_set, num_boost_round=2000,
                     valid_sets=[val_set], callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def train_xgb_v2(X_train, y_train, X_val, y_val, seed=SEED):
    """XGBoost with more aggressive params."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'max_depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'min_child_weight': 5,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'seed': seed,
        'n_jobs': -1,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=2000,
                     evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=False)
    return model


def train_cat_v2(X_train, y_train, X_val, y_val, seed=SEED):
    """CatBoost with more aggressive params."""
    model = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=8,
                               l2_leaf_reg=3, random_seed=seed, verbose=False,
                               early_stopping_rounds=100, task_type='CPU')
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model


def rank_average(preds):
    n = len(preds[0])
    return np.mean([rankdata(p) / n for p in preds], axis=0)


def create_proxy_split(labeled_df, seed=SEED):
    public, private = train_test_split(labeled_df, test_size=0.5,
                                       stratify=labeled_df['is_cheating'], random_state=seed)
    return public.reset_index(drop=True), private.reset_index(drop=True)


def evaluate_proxy(oof_df, public_df, private_df, pred_col='prediction'):
    pub_merged = public_df.merge(oof_df[['user_hash', pred_col]], on='user_hash', how='left')
    priv_merged = private_df.merge(oof_df[['user_hash', pred_col]], on='user_hash', how='left')
    
    pub_result = metric.evaluate(pub_merged['is_cheating'].values, pub_merged[pred_col].values, n_thresholds=N_THRESHOLDS)
    priv_result = metric.evaluate(priv_merged['is_cheating'].values, priv_merged[pred_col].values, n_thresholds=N_THRESHOLDS)
    
    return {
        'public_cost': pub_result.best_cost,
        'private_cost': priv_result.best_cost,
    }


def run_advanced_experiment(labeled_df, neighbors, degree, exp_name="adv_exp"):
    """Run advanced experiment with improved features."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*60}")
    
    y = labeled_df['is_cheating'].values
    users = labeled_df['user_hash'].values
    n = len(labeled_df)
    
    public_df, private_df = create_proxy_split(labeled_df)
    print(f"Proxy: public={len(public_df)}, private={len(private_df)}")
    
    oof_lgb = np.zeros(n)
    oof_xgb = np.zeros(n)
    oof_cat = np.zeros(n)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(labeled_df, y)):
        print(f"\nFold {fold+1}/{N_FOLDS} | Train: {len(train_idx)} | Val: {len(val_idx)}")
        
        train_labels = dict(zip(users[train_idx], y[train_idx]))
        
        X = create_advanced_features(labeled_df, neighbors, degree, train_labels, None)
        X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"  Features: {X.shape[1]}")
        
        lgb_model = train_lgb_v2(X_train, y_train, X_val, y_val)
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        
        xgb_model = train_xgb_v2(X_train, y_train, X_val, y_val)
        oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(X_val))
        
        cat_model = train_cat_v2(X_train, y_train, X_val, y_val)
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        
        for name, preds in [('LGB', oof_lgb[val_idx]), ('XGB', oof_xgb[val_idx]), ('CAT', oof_cat[val_idx])]:
            r = metric.evaluate(y_val, preds, n_thresholds=N_THRESHOLDS)
            print(f"    {name}: cost={r.best_cost:,}")
    
    blend = (oof_lgb + oof_xgb + oof_cat) / 3
    rank_blend = rank_average([oof_lgb, oof_xgb, oof_cat])
    
    # Weighted blend (give more weight to best model)
    lgb_cost = metric.evaluate(y, oof_lgb, n_thresholds=N_THRESHOLDS).best_cost
    xgb_cost = metric.evaluate(y, oof_xgb, n_thresholds=N_THRESHOLDS).best_cost
    cat_cost = metric.evaluate(y, oof_cat, n_thresholds=N_THRESHOLDS).best_cost
    
    # Inverse cost weighting
    inv_costs = 1.0 / np.array([lgb_cost, xgb_cost, cat_cost])
    weights = inv_costs / inv_costs.sum()
    weighted_blend = oof_lgb * weights[0] + oof_xgb * weights[1] + oof_cat * weights[2]
    
    oof_df = pd.DataFrame({
        'user_hash': users,
        'is_cheating': y,
        'pred_lgb': oof_lgb,
        'pred_xgb': oof_xgb,
        'pred_cat': oof_cat,
        'pred_blend': blend,
        'pred_rank': rank_blend,
        'pred_weighted': weighted_blend,
    })
    
    print(f"\n{'='*60}")
    print("OOF RESULTS")
    print(f"{'='*60}")
    
    results = {}
    for name, col in [('LGB', 'pred_lgb'), ('XGB', 'pred_xgb'), ('CAT', 'pred_cat'),
                      ('BLEND', 'pred_blend'), ('RANK', 'pred_rank'), ('WEIGHTED', 'pred_weighted')]:
        r = metric.evaluate(y, oof_df[col].values, n_thresholds=N_THRESHOLDS)
        results[name] = r.best_cost
        print(f"{name}: cost={r.best_cost:,} | t_low={r.t_low:.3f} | t_high={r.t_high:.3f}")
    
    # Find best
    best_name = min(results.keys(), key=lambda k: results[k])
    best_col = f'pred_{best_name.lower()}'
    
    print(f"\nBest: {best_name} with cost={results[best_name]:,}")
    
    # Proxy evaluation
    print(f"\nPROXY EVALUATION")
    proxy_results = evaluate_proxy(oof_df.rename(columns={best_col: 'prediction'}), public_df, private_df)
    print(f"  Public:  {proxy_results['public_cost']:,}")
    print(f"  Private: {proxy_results['private_cost']:,}")
    
    stability = 100 * (proxy_results['public_cost'] - proxy_results['private_cost']) / proxy_results['private_cost']
    print(f"  Stability: {stability:.1f}%")
    
    # Log
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    oof_df.to_csv(os.path.join(ARTIFACTS_DIR, f"oof_{exp_id}.csv"), index=False)
    
    log_row = {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'cv_cost': results[best_name],
        'public_cost': proxy_results['public_cost'],
        'private_cost': proxy_results['private_cost'],
        'stability_pct': stability,
        'best_model': best_name,
        'timestamp': datetime.now().isoformat(),
    }
    
    log_path = os.path.join(ARTIFACTS_DIR, "exp_log.csv")
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_row])
    log_df.to_csv(log_path, index=False)
    
    print(f"\nExperiment logged: {exp_id}")
    
    return oof_df, results, proxy_results, best_col


def train_and_submit(labeled_df, test_df, neighbors, degree, oof_preds=None, sample_sub=None):
    """Train final and create submission."""
    print(f"\n{'='*60}")
    print("FINAL TRAINING")
    print(f"{'='*60}")
    
    y = labeled_df['is_cheating'].values
    all_labels = dict(zip(labeled_df['user_hash'], y))
    
    X_train = create_advanced_features(labeled_df, neighbors, degree, all_labels, oof_preds)
    X_test = create_advanced_features(test_df, neighbors, degree, all_labels, oof_preds)
    X_test = X_test[X_train.columns]
    
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    
    lgb_model = train_lgb_v2(X_train.values, y, X_train.values, y)
    xgb_model = train_xgb_v2(X_train.values, y, X_train.values, y)
    cat_model = train_cat_v2(X_train.values, y, X_train.values, y)
    
    pred_lgb = lgb_model.predict(X_test.values)
    pred_xgb = xgb_model.predict(xgb.DMatrix(X_test.values))
    pred_cat = cat_model.predict_proba(X_test.values)[:, 1]
    
    pred_blend = (pred_lgb + pred_xgb + pred_cat) / 3
    
    print(f"Pred range: [{pred_blend.min():.4f}, {pred_blend.max():.4f}]")
    
    if sample_sub is not None:
        sub = pd.DataFrame({'user_hash': test_df['user_hash'], 'prediction': pred_blend})
        sub = sample_sub[['user_hash']].merge(sub, on='user_hash', how='left')
        sub.to_csv("submission.csv", index=False)
        print(f"Submission saved: submission.csv ({len(sub):,} rows)")
    
    return pred_blend


def main():
    print("=" * 60)
    print("MERCOR - ADVANCED OPTIMIZATION")
    print("=" * 60)
    
    train, test, graph, sample_sub = load_data()
    labeled, unlabeled = split_data(train)
    
    print(f"Labeled: {len(labeled):,} | Unlabeled: {len(unlabeled):,} | Test: {len(test):,}")
    
    all_users = set(train['user_hash'].tolist() + test['user_hash'].tolist())
    neighbors, degree = build_graph(graph, all_users)
    print(f"Graph: {len(neighbors):,} nodes | Max degree: {max(degree.values())}")
    
    # Run advanced experiment
    oof_df, results, proxy_results, best_col = run_advanced_experiment(
        labeled, neighbors, degree, "advanced_v1"
    )
    
    # Train and submit
    oof_preds = dict(zip(oof_df['user_hash'], oof_df[best_col]))
    train_and_submit(labeled, test, neighbors, degree, oof_preds, sample_sub)
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Best Model: {min(results.keys(), key=lambda k: results[k])}")
    print(f"CV Cost: {min(results.values()):,}")
    print(f"Public Proxy: {proxy_results['public_cost']:,}")
    print(f"Private Proxy: {proxy_results['private_cost']:,}")


if __name__ == "__main__":
    main()
