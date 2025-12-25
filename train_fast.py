"""
Mercor Cheating Detection - Fast Experiment Runner

Faster training with reduced hyperparameters for quicker iteration.
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
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


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


def create_features(df, neighbors, degree, train_labels=None, train_preds=None):
    """Create all features."""
    feature_cols = [f"feature_{i:03d}" for i in range(1, 19)]
    
    # Tabular features
    feats = df[feature_cols].copy()
    for col in feature_cols:
        feats[f"{col}_miss"] = feats[col].isna().astype(int)
    
    feats['na_count'] = df[feature_cols].isna().sum(axis=1)
    feats['row_mean'] = df[feature_cols].mean(axis=1)
    feats['row_std'] = df[feature_cols].std(axis=1)
    feats['row_min'] = df[feature_cols].min(axis=1)
    feats['row_max'] = df[feature_cols].max(axis=1)
    feats['row_range'] = feats['row_max'] - feats['row_min']
    feats['row_sum'] = df[feature_cols].sum(axis=1)
    
    # Log transforms
    for col in ['feature_010', 'feature_015', 'feature_016']:
        val = df[col].fillna(0)
        feats[f'{col}_log'] = np.log1p(np.abs(val)) * np.sign(val)
    
    # Ratios
    feats['ratio_018_017'] = df['feature_018'].fillna(0) / (df['feature_017'].fillna(0) + 0.01)
    
    # Graph features
    feats['degree'] = df['user_hash'].map(degree).fillna(0)
    feats['log_degree'] = np.log1p(feats['degree'])
    feats['is_isolated'] = (feats['degree'] == 0).astype(int)
    
    # Neighbor stats
    n_stats = []
    for user in df['user_hash']:
        ns = neighbors.get(user, [])
        if ns:
            ndeg = [degree.get(n, 0) for n in ns]
            n_stats.append({
                'n_mean': np.mean(ndeg),
                'n_std': np.std(ndeg) if len(ndeg) > 1 else 0,
                'n_max': np.max(ndeg),
                'n_count': len(ns),
            })
        else:
            n_stats.append({'n_mean': 0, 'n_std': 0, 'n_max': 0, 'n_count': 0})
    
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
                    label_feats.append({
                        'n_labeled': len(labeled_ns),
                        'cheat_rate': np.mean(labeled_ns),
                    })
                else:
                    label_feats.append({'n_labeled': 0, 'cheat_rate': -1})
            else:
                label_feats.append({'n_labeled': 0, 'cheat_rate': -1})
        
        ldf = pd.DataFrame(label_feats, index=df.index)
        feats['n_labeled'] = ldf['n_labeled']
        feats['cheat_rate'] = ldf['cheat_rate']
        feats['labeled_ratio'] = ldf['n_labeled'] / (feats['degree'] + 1)
    
    # Fold-safe prediction propagation
    if train_preds:
        pred_feats = []
        for user in df['user_hash']:
            ns = neighbors.get(user, [])
            if ns:
                npreds = [train_preds.get(n) for n in ns if n in train_preds]
                if npreds:
                    pred_feats.append({
                        'pred_mean': np.mean(npreds),
                        'pred_max': np.max(npreds),
                    })
                else:
                    pred_feats.append({'pred_mean': -1, 'pred_max': -1})
            else:
                pred_feats.append({'pred_mean': -1, 'pred_max': -1})
        
        pdf = pd.DataFrame(pred_feats, index=df.index)
        feats['pred_mean'] = pdf['pred_mean']
        feats['pred_max'] = pdf['pred_max']
    
    feats = feats.fillna(-999)
    return feats


def train_lgb(X_train, y_train, X_val, y_val, seed=SEED):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 63,
        'learning_rate': 0.05,
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
    model = lgb.train(params, train_set, num_boost_round=1500,
                     valid_sets=[val_set], callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def train_xgb(X_train, y_train, X_val, y_val, seed=SEED):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'max_depth': 7,
        'learning_rate': 0.05,
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
    model = xgb.train(params, dtrain, num_boost_round=1500,
                     evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=False)
    return model


def train_cat(X_train, y_train, X_val, y_val, seed=SEED):
    model = CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=7,
                               l2_leaf_reg=5, random_seed=seed, verbose=False,
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
        'public_score': pub_result.best_score,
        'private_cost': priv_result.best_cost,
        'private_score': priv_result.best_score,
    }


def run_experiment(labeled_df, neighbors, degree, exp_name="exp"):
    """Run CV experiment with logging."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*60}")
    
    y = labeled_df['is_cheating'].values
    users = labeled_df['user_hash'].values
    n = len(labeled_df)
    
    # Create proxy split
    public_df, private_df = create_proxy_split(labeled_df)
    print(f"Proxy: public={len(public_df)}, private={len(private_df)}")
    
    # OOF arrays
    oof_lgb = np.zeros(n)
    oof_xgb = np.zeros(n)
    oof_cat = np.zeros(n)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(labeled_df, y)):
        print(f"\nFold {fold+1}/{N_FOLDS} | Train: {len(train_idx)} | Val: {len(val_idx)}")
        
        train_labels = dict(zip(users[train_idx], y[train_idx]))
        
        X = create_features(labeled_df, neighbors, degree, train_labels, None)
        X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"  Features: {X.shape[1]}")
        
        lgb_model = train_lgb(X_train, y_train, X_val, y_val)
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        
        xgb_model = train_xgb(X_train, y_train, X_val, y_val)
        oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(X_val))
        
        cat_model = train_cat(X_train, y_train, X_val, y_val)
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        
        for name, preds in [('LGB', oof_lgb[val_idx]), ('XGB', oof_xgb[val_idx]), ('CAT', oof_cat[val_idx])]:
            r = metric.evaluate(y_val, preds, n_thresholds=N_THRESHOLDS)
            print(f"    {name}: cost={r.best_cost:,}")
    
    # Create OOF df
    blend = (oof_lgb + oof_xgb + oof_cat) / 3
    rank_blend = rank_average([oof_lgb, oof_xgb, oof_cat])
    
    oof_df = pd.DataFrame({
        'user_hash': users,
        'is_cheating': y,
        'pred_lgb': oof_lgb,
        'pred_xgb': oof_xgb,
        'pred_cat': oof_cat,
        'pred_blend': blend,
        'pred_rank': rank_blend,
    })
    
    # Evaluate
    print(f"\n{'='*60}")
    print("OOF RESULTS")
    print(f"{'='*60}")
    
    results = {}
    for name, col in [('LGB', 'pred_lgb'), ('XGB', 'pred_xgb'), ('CAT', 'pred_cat'),
                      ('BLEND', 'pred_blend'), ('RANK', 'pred_rank')]:
        r = metric.evaluate(y, oof_df[col].values, n_thresholds=N_THRESHOLDS)
        results[name] = r.best_cost
        print(f"{name}: cost={r.best_cost:,} | t_low={r.t_low:.3f} | t_high={r.t_high:.3f}")
    
    # Proxy evaluation
    print(f"\nPROXY EVALUATION")
    best_col = min(['pred_blend', 'pred_rank'], key=lambda c: metric.evaluate(y, oof_df[c].values, n_thresholds=N_THRESHOLDS).best_cost)
    
    proxy_results = evaluate_proxy(oof_df.rename(columns={best_col: 'prediction'}), public_df, private_df)
    print(f"  Public:  {proxy_results['public_cost']:,}")
    print(f"  Private: {proxy_results['private_cost']:,}")
    
    stability = 100 * (proxy_results['public_cost'] - proxy_results['private_cost']) / proxy_results['private_cost']
    print(f"  Stability: {stability:.1f}%")
    
    # Save
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    oof_df.to_csv(os.path.join(ARTIFACTS_DIR, f"oof_{exp_id}.csv"), index=False)
    
    # Log
    log_row = {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'cv_cost': results['BLEND'],
        'public_cost': proxy_results['public_cost'],
        'private_cost': proxy_results['private_cost'],
        'stability_pct': stability,
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
    
    return oof_df, results, proxy_results


def train_and_submit(labeled_df, test_df, neighbors, degree, oof_preds=None):
    """Train final models and create submission."""
    print(f"\n{'='*60}")
    print("FINAL TRAINING")
    print(f"{'='*60}")
    
    y = labeled_df['is_cheating'].values
    all_labels = dict(zip(labeled_df['user_hash'], y))
    
    X_train = create_features(labeled_df, neighbors, degree, all_labels, oof_preds)
    X_test = create_features(test_df, neighbors, degree, all_labels, oof_preds)
    X_test = X_test[X_train.columns]
    
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    
    lgb_model = train_lgb(X_train.values, y, X_train.values, y)
    xgb_model = train_xgb(X_train.values, y, X_train.values, y)
    cat_model = train_cat(X_train.values, y, X_train.values, y)
    
    pred_lgb = lgb_model.predict(X_test.values)
    pred_xgb = xgb_model.predict(xgb.DMatrix(X_test.values))
    pred_cat = cat_model.predict_proba(X_test.values)[:, 1]
    
    pred_blend = (pred_lgb + pred_xgb + pred_cat) / 3
    pred_rank = rank_average([pred_lgb, pred_xgb, pred_cat])
    
    print(f"Pred ranges: blend=[{pred_blend.min():.4f}, {pred_blend.max():.4f}]")
    
    return {'blend': pred_blend, 'rank': pred_rank}


def main():
    print("=" * 60)
    print("MERCOR CHEATING DETECTION - FAST RUNNER")
    print("=" * 60)
    
    train, test, graph, sample_sub = load_data()
    labeled, unlabeled = split_data(train)
    
    print(f"Labeled: {len(labeled):,} | Unlabeled: {len(unlabeled):,} | Test: {len(test):,}")
    
    all_users = set(train['user_hash'].tolist() + test['user_hash'].tolist())
    neighbors, degree = build_graph(graph, all_users)
    print(f"Graph: {len(neighbors):,} nodes | Max degree: {max(degree.values())}")
    
    # Run experiment
    oof_df, cv_results, proxy_results = run_experiment(labeled, neighbors, degree, "fast_baseline")
    
    # Train and submit
    best_col = 'pred_blend'
    oof_preds = dict(zip(oof_df['user_hash'], oof_df[best_col]))
    
    test_preds = train_and_submit(labeled, test, neighbors, degree, oof_preds)
    
    # Create submission
    sub = pd.DataFrame({'user_hash': test['user_hash'], 'prediction': test_preds['blend']})
    sub = sample_sub[['user_hash']].merge(sub, on='user_hash', how='left')
    sub.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved: submission.csv ({len(sub):,} rows)")
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"CV Cost (BLEND): {cv_results['BLEND']:,}")
    print(f"Public Proxy:    {proxy_results['public_cost']:,}")
    print(f"Private Proxy:   {proxy_results['private_cost']:,}")


if __name__ == "__main__":
    main()
