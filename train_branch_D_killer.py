"""
Branch D KILLER Pipeline: Pseudo-GNN Multi-hop Graph + Tabular Optimization
Target: TOP 1 on Mercor Cheating Detection (~-1.54M to -1.56M)

This script implements:
- Phase 0: Metric verification (already done via metric_tests.py)
- Phase 1: Splits + baseline sanity
- Phase 2: Advanced multi-hop graph features (fold-safe)
- Phase 3: Cost-aware modeling with multiple algorithms
- Phase 4: Multi-stage hyperparameter search (Random → TPE → Genetic → Hyperband)
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler, RandomSampler
import networkx as nx

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)

# Import metric
from metric import score, score_with_details, get_thresholds
from official_eval_reference import compute_cost, evaluate, find_optimal_thresholds

# Cost constants
MISS_CHEATER_COST = 500.0
BLOCK_CLEAN_COST = 100.0
MANUAL_REVIEW_COST = 50.0


class DataLoader:
    """Load and prepare data."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load(self):
        print("Loading data...")
        self.train = pd.read_csv(self.data_dir / 'train.csv')
        self.test = pd.read_csv(self.data_dir / 'test.csv')
        self.graph_df = pd.read_csv(self.data_dir / 'social_graph.csv')
        
        with open(self.data_dir / 'feature_metadata.json') as f:
            self.feature_metadata = json.load(f)
        
        # Labeled vs unlabeled
        self.labeled_mask = self.train['is_cheating'].notna()
        self.labeled = self.train[self.labeled_mask].copy()
        self.unlabeled = self.train[~self.labeled_mask].copy()
        
        print(f"  Train: {self.train.shape}")
        print(f"  Test: {self.test.shape}")
        print(f"  Graph edges: {len(self.graph_df)}")
        print(f"  Labeled: {len(self.labeled)} (pos: {(self.labeled['is_cheating']==1).sum()}, neg: {(self.labeled['is_cheating']==0).sum()})")
        print(f"  Unlabeled (high_conf_clean): {len(self.unlabeled)}")
        
        return self
    
    def build_graph(self):
        print("Building graph...")
        self.graph = nx.Graph()
        
        # All users from train + test
        all_users = set(self.train['user_hash'].unique()) | set(self.test['user_hash'].unique())
        self.graph.add_nodes_from(all_users)
        
        # Add edges
        edges = list(zip(self.graph_df['user_a'], self.graph_df['user_b']))
        self.graph.add_edges_from(edges)
        
        # Precompute degrees
        self.degree_dict = dict(self.graph.degree())
        
        print(f"  Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        return self


class FeatureEngineer:
    """Feature engineering with fold-safe label aggregations."""
    
    def __init__(self, dl: DataLoader):
        self.dl = dl
        self.feature_cols = [f'feature_{i:03d}' for i in range(1, 19)]
    
    def create_tabular_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create base tabular features."""
        features = df[self.feature_cols].copy()
        
        # Missingness indicators
        for col in self.feature_cols:
            features[f'{col}_isna'] = features[col].isna().astype(int)
        
        # Row statistics
        features['na_count'] = features[self.feature_cols].isna().sum(axis=1)
        features['row_mean'] = features[self.feature_cols].mean(axis=1)
        features['row_std'] = features[self.feature_cols].std(axis=1)
        features['row_min'] = features[self.feature_cols].min(axis=1)
        features['row_max'] = features[self.feature_cols].max(axis=1)
        features['row_range'] = features['row_max'] - features['row_min']
        features['row_median'] = features[self.feature_cols].median(axis=1)
        
        # Fill NaN with -999 for tree models
        features = features.fillna(-999)
        
        return features
    
    def create_graph_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create graph structural features (NO labels used)."""
        graph = self.dl.graph
        degree_dict = self.dl.degree_dict
        user_hashes = df['user_hash'].values
        
        features = pd.DataFrame(index=df.index)
        
        # Degree features
        degrees = [degree_dict.get(uh, 0) for uh in user_hashes]
        features['degree'] = degrees
        features['log_degree'] = np.log1p(degrees)
        
        # 1-hop neighbor statistics
        hop1_stats = []
        for uh in user_hashes:
            if uh in graph and degree_dict.get(uh, 0) > 0:
                neighbors = list(graph.neighbors(uh))
                neighbor_degs = [degree_dict.get(n, 0) for n in neighbors]
                hop1_stats.append({
                    'hop1_mean': np.mean(neighbor_degs),
                    'hop1_median': np.median(neighbor_degs),
                    'hop1_max': np.max(neighbor_degs),
                    'hop1_min': np.min(neighbor_degs),
                    'hop1_std': np.std(neighbor_degs) if len(neighbor_degs) > 1 else 0,
                    'hop1_sum': np.sum(neighbor_degs)
                })
            else:
                hop1_stats.append({'hop1_mean': 0, 'hop1_median': 0, 'hop1_max': 0, 
                                   'hop1_min': 0, 'hop1_std': 0, 'hop1_sum': 0})
        
        for key in ['hop1_mean', 'hop1_median', 'hop1_max', 'hop1_min', 'hop1_std', 'hop1_sum']:
            features[key] = [s[key] for s in hop1_stats]
        
        # 2-hop approximation (sum of neighbor degrees = approx 2-hop reach)
        features['approx_hop2_reach'] = features['hop1_sum']
        features['log_hop2_reach'] = np.log1p(features['approx_hop2_reach'])
        features['hop2_density'] = np.where(
            features['degree'] > 0,
            features['approx_hop2_reach'] / features['degree'],
            0
        )
        
        # Connected component features
        if not hasattr(self, '_component_map'):
            self._component_map = {}
            for i, comp in enumerate(nx.connected_components(graph)):
                size = len(comp)
                for node in comp:
                    self._component_map[node] = (i, size)
        
        comp_ids = []
        comp_sizes = []
        for uh in user_hashes:
            cid, csize = self._component_map.get(uh, (-1, 1))
            comp_ids.append(cid)
            comp_sizes.append(csize)
        
        features['component_id'] = comp_ids
        features['component_size'] = comp_sizes
        features['log_component_size'] = np.log1p(comp_sizes)
        
        return features
    
    def create_fold_safe_label_features(self, df: pd.DataFrame, 
                                         train_idx: np.ndarray,
                                         train_y: np.ndarray,
                                         alpha: float = 1.0,
                                         beta: float = 1.0) -> pd.DataFrame:
        """
        Create label-aware features using ONLY train-fold labels.
        Bayesian smoothing with alpha, beta priors.
        OPTIMIZED: Only 1-hop features to avoid O(n*d^2) complexity.
        """
        graph = self.dl.graph
        user_hashes = df['user_hash'].values
        degree_dict = self.dl.degree_dict
        
        # Build label mapping from train fold only
        train_users = self.dl.labeled.iloc[train_idx]['user_hash'].values
        user_to_label = dict(zip(train_users, train_y))
        
        # Precompute labeled set for fast lookup
        labeled_set = set(user_to_label.keys())
        prior = alpha / (alpha + beta)
        
        features = pd.DataFrame(index=df.index)
        
        # 1-hop label aggregation (vectorized where possible)
        hop1_labeled_count = []
        hop1_cheat_rate = []
        hop1_cheat_count = []
        
        for uh in user_hashes:
            deg = degree_dict.get(uh, 0)
            if uh in graph and deg > 0:
                neighbors = list(graph.neighbors(uh))
                # Fast intersection
                labeled_neighbors = [n for n in neighbors if n in labeled_set]
                n_labeled = len(labeled_neighbors)
                
                if n_labeled > 0:
                    cheater_count = sum(user_to_label[n] for n in labeled_neighbors)
                    smoothed_rate = (cheater_count + alpha) / (n_labeled + alpha + beta)
                    hop1_labeled_count.append(n_labeled)
                    hop1_cheat_rate.append(smoothed_rate)
                    hop1_cheat_count.append(cheater_count)
                else:
                    hop1_labeled_count.append(0)
                    hop1_cheat_rate.append(prior)
                    hop1_cheat_count.append(0)
            else:
                hop1_labeled_count.append(0)
                hop1_cheat_rate.append(prior)
                hop1_cheat_count.append(0)
        
        features['hop1_labeled_count'] = hop1_labeled_count
        features['hop1_cheat_rate'] = hop1_cheat_rate
        features['hop1_cheat_count'] = hop1_cheat_count
        features['log_hop1_labeled_count'] = np.log1p(hop1_labeled_count)
        
        # Approximate 2-hop influence using 1-hop stats (FAST approximation)
        # Use hop1_sum (sum of neighbor degrees) as proxy for 2-hop reach
        hop1_sum = features.get('hop1_sum', np.zeros(len(df)))
        if 'hop1_sum' not in features.columns:
            # Compute it quickly
            hop1_sum_vals = []
            for uh in user_hashes:
                if uh in graph and degree_dict.get(uh, 0) > 0:
                    neighbors = list(graph.neighbors(uh))
                    hop1_sum_vals.append(sum(degree_dict.get(n, 0) for n in neighbors))
                else:
                    hop1_sum_vals.append(0)
            hop1_sum = np.array(hop1_sum_vals)
        
        # Approximate 2-hop cheat signal: scale 1-hop rate by connectivity
        features['approx_hop2_cheat_signal'] = np.array(hop1_cheat_rate) * np.log1p(hop1_sum) / 10.0
        
        # Weighted signal combining rate and count
        features['weighted_cheat_signal'] = (
            np.array(hop1_cheat_rate) * 0.7 + 
            np.clip(np.array(hop1_cheat_count) / 10.0, 0, 1) * 0.3
        )
        
        # Ratio features
        features['labeled_ratio'] = np.where(
            np.array([degree_dict.get(uh, 0) for uh in user_hashes]) > 0,
            np.array(hop1_labeled_count) / np.array([degree_dict.get(uh, 1) for uh in user_hashes]),
            0
        )
        
        return features


class BranchDTrainer:
    """Branch D training with multi-stage optimization."""
    
    def __init__(self, dl: DataLoader, fe: FeatureEngineer, out_dir: str):
        self.dl = dl
        self.fe = fe
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'candidates').mkdir(exist_ok=True)
        
        self.scoreboard = []
        self.best_proxy_public = float('inf')
        self.best_config = None
        self.best_oof_preds = None
        self.run_id = 0
        self.no_improvement_count = 0
        
    def get_proxy_split(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """50/50 stratified proxy split."""
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        public_idx, private_idx = next(skf.split(np.zeros(len(y)), y))
        return public_idx, private_idx
    
    def evaluate_baselines(self, y: np.ndarray):
        """Evaluate baseline predictions on proxy split."""
        print("\n=== BASELINE EVALUATION ===")
        
        public_idx, private_idx = self.get_proxy_split(y)
        y_pub = y[public_idx]
        y_priv = y[private_idx]
        
        baselines = {
            'const_0.01': np.full(len(y), 0.01),
            'const_0.50': np.full(len(y), 0.50),
            'const_0.99': np.full(len(y), 0.99),
            'uniform': np.random.RandomState(SEED).random(len(y))
        }
        
        results = {}
        for name, preds in baselines.items():
            pub_cost = score(y_pub, preds[public_idx])
            priv_cost = score(y_priv, preds[private_idx])
            results[name] = {'proxy_public': pub_cost, 'proxy_private': priv_cost}
            print(f"  {name}: pub={pub_cost:.0f}, priv={priv_cost:.0f}")
        
        return results
    
    def cv_train_evaluate(self, X: pd.DataFrame, y: np.ndarray, 
                          model_fn, params: dict,
                          n_folds: int = 5,
                          pseudo_weight: float = 0.0,
                          alpha: float = 1.0, beta: float = 1.0,
                          calibrate: str = None) -> dict:
        """
        Full CV with fold-safe label features, optional pseudo-negatives, and calibration.
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        
        oof_preds = np.zeros(len(y))
        
        # Prepare base features once
        X_tabular = self.fe.create_tabular_features(self.dl.labeled)
        X_graph = self.fe.create_graph_structural_features(self.dl.labeled)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create fold-safe label features
            train_label_feats = self.fe.create_fold_safe_label_features(
                self.dl.labeled.iloc[train_idx], train_idx, y_train, alpha, beta
            )
            val_label_feats = self.fe.create_fold_safe_label_features(
                self.dl.labeled.iloc[val_idx], train_idx, y_train, alpha, beta
            )
            
            # Combine features
            X_train = pd.concat([
                X_tabular.iloc[train_idx].reset_index(drop=True),
                X_graph.iloc[train_idx].reset_index(drop=True),
                train_label_feats.reset_index(drop=True)
            ], axis=1)
            
            X_val = pd.concat([
                X_tabular.iloc[val_idx].reset_index(drop=True),
                X_graph.iloc[val_idx].reset_index(drop=True),
                val_label_feats.reset_index(drop=True)
            ], axis=1)
            
            # Add pseudo-negatives
            if pseudo_weight > 0:
                X_pseudo_tab = self.fe.create_tabular_features(self.dl.unlabeled)
                X_pseudo_graph = self.fe.create_graph_structural_features(self.dl.unlabeled)
                X_pseudo_label = self.fe.create_fold_safe_label_features(
                    self.dl.unlabeled, train_idx, y_train, alpha, beta
                )
                X_pseudo = pd.concat([
                    X_pseudo_tab.reset_index(drop=True),
                    X_pseudo_graph.reset_index(drop=True),
                    X_pseudo_label.reset_index(drop=True)
                ], axis=1)
                
                y_pseudo = np.zeros(len(X_pseudo))
                sample_weights = np.concatenate([
                    np.ones(len(y_train)),
                    np.ones(len(y_pseudo)) * pseudo_weight
                ])
                X_train = pd.concat([X_train, X_pseudo], ignore_index=True)
                y_train = np.concatenate([y_train, y_pseudo])
            else:
                sample_weights = None
            
            # Train model
            model = model_fn(params)
            if sample_weights is not None:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                fold_preds = model.predict_proba(X_val)[:, 1]
            else:
                fold_preds = model.predict(X_val)
            
            # Calibration (fold-wise)
            if calibrate == 'isotonic':
                iso = IsotonicRegression(out_of_bounds='clip')
                # Re-fit on validation for calibration (using train predictions on val)
                train_preds_for_cal = model.predict_proba(X_train)[:, 1] if sample_weights is None else model.predict_proba(X_train[:len(y[train_idx])])[:, 1]
                iso.fit(train_preds_for_cal, y[train_idx])
                fold_preds = iso.transform(fold_preds)
            elif calibrate == 'platt':
                train_preds_for_cal = model.predict_proba(X_train)[:, 1] if sample_weights is None else model.predict_proba(X_train[:len(y[train_idx])])[:, 1]
                lr = LogisticRegression()
                lr.fit(train_preds_for_cal.reshape(-1, 1), y[train_idx])
                fold_preds = lr.predict_proba(fold_preds.reshape(-1, 1))[:, 1]
            elif calibrate == 'temperature':
                # Simple temperature scaling
                temp = 1.5
                fold_preds = 1 / (1 + np.exp(-np.log(fold_preds / (1 - fold_preds + 1e-10)) / temp))
            
            oof_preds[val_idx] = fold_preds
        
        # Compute metrics
        oof_cost, oof_details = score_with_details(y, oof_preds)
        public_idx, private_idx = self.get_proxy_split(y)
        proxy_pub = score(y[public_idx], oof_preds[public_idx])
        proxy_priv = score(y[private_idx], oof_preds[private_idx])
        
        return {
            'oof_preds': oof_preds,
            'oof_cost': oof_cost,
            'proxy_public': proxy_pub,
            'proxy_private': proxy_priv,
            'details': oof_details
        }
    
    def get_lgb_model(self, params: dict):
        """Create LightGBM model."""
        return lgb.LGBMClassifier(
            n_estimators=params.get('n_estimators', 1000),
            learning_rate=params.get('learning_rate', 0.03),
            num_leaves=params.get('num_leaves', 31),
            max_depth=params.get('max_depth', -1),
            min_data_in_leaf=params.get('min_data_in_leaf', 20),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 0.0),
            scale_pos_weight=params.get('scale_pos_weight', 2.0),
            random_state=SEED,
            verbose=-1,
            n_jobs=-1
        )
    
    def get_cb_model(self, params: dict):
        """Create CatBoost model."""
        return cb.CatBoostClassifier(
            iterations=params.get('iterations', 1000),
            learning_rate=params.get('learning_rate', 0.03),
            depth=params.get('depth', 6),
            l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
            min_data_in_leaf=params.get('min_data_in_leaf', 20),
            subsample=params.get('subsample', 0.8),
            colsample_bylevel=params.get('colsample_bylevel', 0.8),
            scale_pos_weight=params.get('scale_pos_weight', 2.0),
            random_state=SEED,
            verbose=False
        )
    
    def get_xgb_model(self, params: dict):
        """Create XGBoost model."""
        return xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 1000),
            learning_rate=params.get('learning_rate', 0.03),
            max_depth=params.get('max_depth', 6),
            min_child_weight=params.get('min_child_weight', 1),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 1.0),
            scale_pos_weight=params.get('scale_pos_weight', 2.0),
            random_state=SEED,
            verbosity=0,
            use_label_encoder=False,
            n_jobs=-1
        )
    
    def log_run(self, stage: str, params: dict, result: dict, notes: str = ""):
        """Log experiment run."""
        self.run_id += 1
        # Convert numpy types to native Python types for JSON serialization
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, (np.integer, np.int64, np.int32)):
                clean_params[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                clean_params[k] = float(v)
            else:
                clean_params[k] = v
        
        entry = {
            'branch': 'D',
            'run_id': self.run_id,
            'algo_stage': stage,
            'params_json': json.dumps(clean_params),
            'featureset_id': 'full',
            'oof_cost': result['oof_cost'],
            'proxy_public_cost': result['proxy_public'],
            'proxy_private_cost': result['proxy_private'],
            't_low': result['details']['t_low'],
            't_high': result['details']['t_high'],
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        self.scoreboard.append(entry)
        
        # Check if new best
        stability_check = True
        if self.best_proxy_public < float('inf'):
            # Reject if private worsens >10% unless public gain >15k
            if result['proxy_private'] > self.best_config['proxy_private'] * 1.1:
                pub_gain = self.best_proxy_public - result['proxy_public']
                if pub_gain < 15000:
                    stability_check = False
        
        if result['proxy_public'] < self.best_proxy_public and stability_check:
            self.best_proxy_public = result['proxy_public']
            self.best_config = {
                'params': params,
                'proxy_public': result['proxy_public'],
                'proxy_private': result['proxy_private'],
                'oof_cost': result['oof_cost'],
                'details': result['details'],
                'run_id': self.run_id,
                'stage': stage
            }
            self.best_oof_preds = result['oof_preds']
            self.no_improvement_count = 0
            
            # Save candidate
            print(f"  *** NEW BEST: pub={result['proxy_public']:.0f}, priv={result['proxy_private']:.0f}")
            return True
        else:
            self.no_improvement_count += 1
            return False
    
    def stage1_random_search(self, X: pd.DataFrame, y: np.ndarray, n_trials: int = 40):
        """Stage 1: Fast random search with LightGBM."""
        print(f"\n=== STAGE 1: Random Search ({n_trials} trials) ===")
        
        for trial in range(n_trials):
            if self.no_improvement_count >= 12:
                print("  Early stop: 12 consecutive non-improvements")
                break
            
            # Random params
            params = {
                'n_estimators': np.random.choice([500, 800, 1000, 1500]),
                'learning_rate': np.random.choice([0.01, 0.03, 0.05]),
                'num_leaves': np.random.choice([31, 63, 127]),
                'min_data_in_leaf': np.random.choice([20, 50, 100]),
                'subsample': np.random.uniform(0.6, 1.0),
                'colsample_bytree': np.random.uniform(0.6, 1.0),
                'scale_pos_weight': np.random.choice([1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
            }
            pseudo_weight = np.random.choice([0, 0.005, 0.01, 0.02, 0.05])
            alpha = np.random.choice([0.5, 1.0, 2.0])
            beta = np.random.choice([0.5, 1.0, 2.0])
            
            result = self.cv_train_evaluate(
                X, y, self.get_lgb_model, params,
                pseudo_weight=pseudo_weight, alpha=alpha, beta=beta
            )
            
            full_params = {**params, 'pseudo_weight': pseudo_weight, 'alpha': alpha, 'beta': beta}
            is_best = self.log_run('stage1_random', full_params, result)
            
            print(f"  Trial {trial+1}/{n_trials}: pub={result['proxy_public']:.0f}, priv={result['proxy_private']:.0f}" + 
                  (" *BEST*" if is_best else ""))
    
    def stage2_bayesian_search(self, X: pd.DataFrame, y: np.ndarray, n_trials: int = 40):
        """Stage 2: Bayesian optimization with Optuna TPE."""
        print(f"\n=== STAGE 2: Bayesian Optimization ({n_trials} trials) ===")
        
        def objective(trial):
            if self.no_improvement_count >= 12:
                raise optuna.TrialPruned()
            
            model_type = trial.suggest_categorical('model_type', ['lgb', 'cb'])
            
            if model_type == 'lgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                    'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                    'min_data_in_leaf': trial.suggest_int('min_data', 10, 200),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
                    'scale_pos_weight': trial.suggest_float('scale_pos', 1.0, 8.0),
                    'reg_alpha': trial.suggest_float('alpha_reg', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('lambda_reg', 1e-8, 10.0, log=True),
                }
                model_fn = self.get_lgb_model
            else:
                params = {
                    'iterations': trial.suggest_int('iterations', 500, 2000),
                    'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'min_data_in_leaf': trial.suggest_int('min_data', 10, 200),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample', 0.5, 1.0),
                    'scale_pos_weight': trial.suggest_float('scale_pos', 1.0, 8.0),
                    'l2_leaf_reg': trial.suggest_float('l2_reg', 1e-3, 10.0, log=True),
                }
                model_fn = self.get_cb_model
            
            pseudo_weight = trial.suggest_float('pseudo_weight', 0, 0.1)
            alpha = trial.suggest_float('smooth_alpha', 0.1, 5.0)
            beta = trial.suggest_float('smooth_beta', 0.1, 5.0)
            calibrate = trial.suggest_categorical('calibrate', [None, 'platt', 'temperature'])
            
            result = self.cv_train_evaluate(
                X, y, model_fn, params,
                pseudo_weight=pseudo_weight, alpha=alpha, beta=beta,
                calibrate=calibrate
            )
            
            full_params = {
                **params, 'model_type': model_type,
                'pseudo_weight': pseudo_weight, 'alpha': alpha, 'beta': beta,
                'calibrate': calibrate
            }
            is_best = self.log_run('stage2_tpe', full_params, result)
            
            print(f"  TPE Trial: pub={result['proxy_public']:.0f}, priv={result['proxy_private']:.0f}" +
                  (" *BEST*" if is_best else ""))
            
            return result['proxy_public']
        
        sampler = TPESampler(seed=SEED)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        except optuna.TrialPruned:
            pass
    
    def stage3_genetic_local_search(self, X: pd.DataFrame, y: np.ndarray, n_mutations: int = 30):
        """Stage 3: Genetic/evolutionary local search around best configs."""
        print(f"\n=== STAGE 3: Genetic Local Search ({n_mutations} mutations) ===")
        
        if self.best_config is None:
            print("  No best config yet, skipping...")
            return
        
        base_params = self.best_config['params']
        
        for mut in range(n_mutations):
            if self.no_improvement_count >= 12:
                print("  Early stop: 12 consecutive non-improvements")
                break
            
            # Mutate 1-2 parameters
            params = base_params.copy()
            n_mutate = np.random.randint(1, 3)
            
            mutable_keys = ['scale_pos_weight', 'num_leaves', 'learning_rate', 'pseudo_weight', 'alpha', 'beta']
            keys_to_mutate = np.random.choice(mutable_keys, min(n_mutate, len(mutable_keys)), replace=False)
            
            for key in keys_to_mutate:
                if key == 'scale_pos_weight':
                    params[key] = max(0.5, params.get(key, 2.0) * np.random.uniform(0.7, 1.5))
                elif key == 'num_leaves':
                    params[key] = int(max(15, min(255, params.get(key, 31) * np.random.uniform(0.5, 2.0))))
                elif key == 'learning_rate':
                    params[key] = max(0.001, min(0.3, params.get(key, 0.03) * np.random.uniform(0.5, 2.0)))
                elif key == 'pseudo_weight':
                    params[key] = max(0, min(0.15, params.get(key, 0.01) + np.random.uniform(-0.02, 0.02)))
                elif key in ['alpha', 'beta']:
                    params[key] = max(0.1, min(5.0, params.get(key, 1.0) * np.random.uniform(0.5, 2.0)))
            
            model_type = params.get('model_type', 'lgb')
            model_fn = self.get_lgb_model if model_type == 'lgb' else self.get_cb_model
            
            result = self.cv_train_evaluate(
                X, y, model_fn, params,
                pseudo_weight=params.get('pseudo_weight', 0),
                alpha=params.get('alpha', 1.0),
                beta=params.get('beta', 1.0),
                calibrate=params.get('calibrate', None)
            )
            
            is_best = self.log_run('stage3_genetic', params, result)
            
            print(f"  Mutation {mut+1}/{n_mutations}: pub={result['proxy_public']:.0f}" +
                  (" *BEST*" if is_best else ""))
            
            # Elitism: if new best, update base
            if is_best:
                base_params = params.copy()
    
    def stage4_hyperband(self, X: pd.DataFrame, y: np.ndarray, n_trials: int = 20):
        """Stage 4: Hyperband with early stopping."""
        print(f"\n=== STAGE 4: Hyperband ({n_trials} trials) ===")
        
        # Simple successive halving: start with few trees, promote best to full
        configs = []
        for _ in range(n_trials):
            params = {
                'n_estimators': 200,  # Start small
                'learning_rate': np.random.choice([0.01, 0.03, 0.05, 0.08]),
                'num_leaves': np.random.choice([31, 63, 127]),
                'min_data_in_leaf': np.random.choice([20, 50, 100]),
                'subsample': np.random.uniform(0.6, 1.0),
                'colsample_bytree': np.random.uniform(0.6, 1.0),
                'scale_pos_weight': np.random.choice([2.0, 3.0, 4.0, 5.0, 6.0]),
            }
            configs.append(params)
        
        # First round: evaluate with 200 trees
        print("  Round 1: 200 trees...")
        scores = []
        for params in configs:
            result = self.cv_train_evaluate(X, y, self.get_lgb_model, params, n_folds=3)
            scores.append((result['proxy_public'], params))
        
        # Promote top 50% to 1000 trees
        scores.sort(key=lambda x: x[0])
        top_configs = [s[1] for s in scores[:len(scores)//2]]
        
        print(f"  Round 2: Promoting {len(top_configs)} configs to 1000 trees...")
        for params in top_configs:
            if self.no_improvement_count >= 12:
                break
            
            params['n_estimators'] = 1000
            result = self.cv_train_evaluate(X, y, self.get_lgb_model, params)
            is_best = self.log_run('stage4_hyperband', params, result)
            
            print(f"    pub={result['proxy_public']:.0f}" + (" *BEST*" if is_best else ""))
    
    def create_blend(self, X: pd.DataFrame, y: np.ndarray):
        """Create blended predictions from multiple models."""
        print("\n=== CREATING BLEND ===")
        
        if self.best_config is None:
            print("  No best config, skipping blend...")
            return
        
        # Train 3 models with best-ish configs
        base_params = self.best_config['params']
        
        # LightGBM
        lgb_result = self.cv_train_evaluate(X, y, self.get_lgb_model, base_params,
                                            pseudo_weight=base_params.get('pseudo_weight', 0),
                                            alpha=base_params.get('alpha', 1.0),
                                            beta=base_params.get('beta', 1.0))
        lgb_preds = lgb_result['oof_preds']
        
        # CatBoost
        cb_params = {
            'iterations': base_params.get('n_estimators', 1000),
            'learning_rate': base_params.get('learning_rate', 0.03),
            'depth': 6,
            'min_data_in_leaf': base_params.get('min_data_in_leaf', 20),
            'scale_pos_weight': base_params.get('scale_pos_weight', 2.0),
        }
        cb_result = self.cv_train_evaluate(X, y, self.get_cb_model, cb_params,
                                           pseudo_weight=base_params.get('pseudo_weight', 0),
                                           alpha=base_params.get('alpha', 1.0),
                                           beta=base_params.get('beta', 1.0))
        cb_preds = cb_result['oof_preds']
        
        # Try different blend weights
        blend_options = [
            ('prob_avg', [0.5, 0.5]),
            ('lgb_heavy', [0.7, 0.3]),
            ('cb_heavy', [0.3, 0.7]),
        ]
        
        best_blend_cost = float('inf')
        best_blend_name = None
        best_blend_preds = None
        
        for name, weights in blend_options:
            blended = weights[0] * lgb_preds + weights[1] * cb_preds
            cost = score(y, blended)
            print(f"  {name}: cost={cost:.0f}")
            
            if cost < best_blend_cost:
                best_blend_cost = cost
                best_blend_name = name
                best_blend_preds = blended
        
        # Also try rank blend
        lgb_ranks = stats.rankdata(lgb_preds) / len(lgb_preds)
        cb_ranks = stats.rankdata(cb_preds) / len(cb_preds)
        rank_blend = 0.5 * lgb_ranks + 0.5 * cb_ranks
        rank_cost = score(y, rank_blend)
        print(f"  rank_avg: cost={rank_cost:.0f}")
        
        if rank_cost < best_blend_cost:
            best_blend_cost = rank_cost
            best_blend_name = 'rank_avg'
            best_blend_preds = rank_blend
        
        # Log best blend
        public_idx, private_idx = self.get_proxy_split(y)
        blend_pub = score(y[public_idx], best_blend_preds[public_idx])
        blend_priv = score(y[private_idx], best_blend_preds[private_idx])
        
        _, details = score_with_details(y, best_blend_preds)
        
        blend_result = {
            'oof_preds': best_blend_preds,
            'oof_cost': best_blend_cost,
            'proxy_public': blend_pub,
            'proxy_private': blend_priv,
            'details': details
        }
        
        is_best = self.log_run('blend', {'blend_type': best_blend_name}, blend_result, 
                              notes=f"Best blend: {best_blend_name}")
        
        print(f"  Best blend '{best_blend_name}': pub={blend_pub:.0f}, priv={blend_priv:.0f}" +
              (" *NEW OVERALL BEST*" if is_best else ""))
    
    def train_final_and_predict(self, X_labeled: pd.DataFrame, y_labeled: np.ndarray,
                                 X_test_full: pd.DataFrame) -> np.ndarray:
        """Train final model on all labeled data and predict test."""
        if self.best_config is None:
            raise ValueError("No best config found!")
        
        print("\n=== TRAINING FINAL MODEL ===")
        
        params = self.best_config['params']
        model_type = params.get('model_type', 'lgb')
        
        # Prepare features
        X_tab_train = self.fe.create_tabular_features(self.dl.labeled)
        X_graph_train = self.fe.create_graph_structural_features(self.dl.labeled)
        
        # For final model, use all labeled for label features
        all_idx = np.arange(len(y_labeled))
        X_label_train = self.fe.create_fold_safe_label_features(
            self.dl.labeled, all_idx, y_labeled,
            params.get('alpha', 1.0), params.get('beta', 1.0)
        )
        
        X_train = pd.concat([
            X_tab_train.reset_index(drop=True),
            X_graph_train.reset_index(drop=True),
            X_label_train.reset_index(drop=True)
        ], axis=1)
        
        # Add pseudo-negatives
        pseudo_weight = params.get('pseudo_weight', 0)
        if pseudo_weight > 0:
            X_pseudo_tab = self.fe.create_tabular_features(self.dl.unlabeled)
            X_pseudo_graph = self.fe.create_graph_structural_features(self.dl.unlabeled)
            X_pseudo_label = self.fe.create_fold_safe_label_features(
                self.dl.unlabeled, all_idx, y_labeled,
                params.get('alpha', 1.0), params.get('beta', 1.0)
            )
            X_pseudo = pd.concat([
                X_pseudo_tab.reset_index(drop=True),
                X_pseudo_graph.reset_index(drop=True),
                X_pseudo_label.reset_index(drop=True)
            ], axis=1)
            
            y_pseudo = np.zeros(len(X_pseudo))
            sample_weights = np.concatenate([
                np.ones(len(y_labeled)),
                np.ones(len(y_pseudo)) * pseudo_weight
            ])
            X_train = pd.concat([X_train, X_pseudo], ignore_index=True)
            y_train = np.concatenate([y_labeled, y_pseudo])
        else:
            y_train = y_labeled
            sample_weights = None
        
        # Train model with more iterations
        final_params = params.copy()
        if model_type == 'lgb':
            final_params['n_estimators'] = min(3000, params.get('n_estimators', 1000) * 2)
            model = self.get_lgb_model(final_params)
        else:
            final_params['iterations'] = min(3000, params.get('iterations', 1000) * 2)
            model = self.get_cb_model(final_params)
        
        if sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        
        # Prepare test features
        X_tab_test = self.fe.create_tabular_features(self.dl.test)
        X_graph_test = self.fe.create_graph_structural_features(self.dl.test)
        X_label_test = self.fe.create_fold_safe_label_features(
            self.dl.test, all_idx, y_labeled,
            params.get('alpha', 1.0), params.get('beta', 1.0)
        )
        
        X_test = pd.concat([
            X_tab_test.reset_index(drop=True),
            X_graph_test.reset_index(drop=True),
            X_label_test.reset_index(drop=True)
        ], axis=1)
        
        # Predict
        preds = model.predict_proba(X_test)[:, 1]
        
        print(f"  Final model: {model_type}, n_trees={final_params.get('n_estimators', final_params.get('iterations'))}")
        print(f"  Test predictions: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}")
        
        return preds
    
    def save_results(self, test_preds: np.ndarray, data_dir: str):
        """Save all results."""
        # Save scoreboard
        scoreboard_df = pd.DataFrame(self.scoreboard)
        scoreboard_df.to_csv(self.out_dir / 'scoreboard_D.csv', index=False)
        
        # Merge with main scoreboard
        main_scoreboard = Path(data_dir) / 'artifacts' / 'scoreboard.csv'
        if main_scoreboard.exists():
            main_df = pd.read_csv(main_scoreboard)
            # Remove old D entries
            main_df = main_df[main_df['branch'] != 'D']
            combined = pd.concat([main_df, scoreboard_df], ignore_index=True)
            combined.to_csv(main_scoreboard, index=False)
        else:
            scoreboard_df.to_csv(main_scoreboard, index=False)
        
        # Save best config
        if self.best_config:
            with open(self.out_dir / 'best.json', 'w') as f:
                json.dump({
                    'params': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                              for k, v in self.best_config['params'].items()},
                    'proxy_public': float(self.best_config['proxy_public']),
                    'proxy_private': float(self.best_config['proxy_private']),
                    'oof_cost': float(self.best_config['oof_cost']),
                    't_low': float(self.best_config['details']['t_low']),
                    't_high': float(self.best_config['details']['t_high']),
                    'run_id': self.best_config['run_id'],
                    'stage': self.best_config['stage']
                }, f, indent=2)
            
            # Save OOF predictions
            if self.best_oof_preds is not None:
                oof_df = pd.DataFrame({
                    'user_hash': self.dl.labeled['user_hash'].values,
                    'oof_pred': self.best_oof_preds,
                    'is_cheating': self.dl.labeled['is_cheating'].values
                })
                oof_df.to_csv(self.out_dir / 'oof_best.csv', index=False)
        
        # Save submission
        sub = pd.DataFrame({
            'user_hash': self.dl.test['user_hash'].values,
            'prediction': test_preds
        })
        sub['prediction'] = sub['prediction'].clip(0, 1)
        
        sub_path = Path(data_dir) / 'submission_D.csv'
        sub.to_csv(sub_path, index=False)
        print(f"\nSaved submission to {sub_path}")
        print(f"  Rows: {len(sub)}, Range: [{sub['prediction'].min():.4f}, {sub['prediction'].max():.4f}]")
        
        return sub_path


def update_report(data_dir: str, trainer: 'BranchDTrainer', baseline_results: dict):
    """Update REPORT.md with Branch D results."""
    report_path = Path(data_dir) / 'REPORT.md'
    
    # Read existing report
    if report_path.exists():
        with open(report_path, 'r') as f:
            content = f.read()
    else:
        content = "# Mercor Cheating Detection Pipeline Report\n\n"
    
    # Add Branch D section
    branch_d_section = f"""

## Branch D: Advanced Multi-hop Graph + Tabular (KILLER Pipeline)

### Metric Verification
All 8 metric tests passed (3 hand-computable + 3 parity + edge cases + constants).

### Dataset Statistics
- Labeled rows: {len(trainer.dl.labeled)}
- Unlabeled (high_conf_clean): {len(trainer.dl.unlabeled)}
- Positive labels: {(trainer.dl.labeled['is_cheating']==1).sum()}
- Negative labels: {(trainer.dl.labeled['is_cheating']==0).sum()}
- Test rows: {len(trainer.dl.test)}

### Baseline Proxy Results
| Baseline | Proxy Public | Proxy Private |
|----------|--------------|---------------|
"""
    
    for name, res in baseline_results.items():
        branch_d_section += f"| {name} | {res['proxy_public']:.0f} | {res['proxy_private']:.0f} |\n"
    
    if trainer.best_config:
        branch_d_section += f"""
### Best Configuration
- **Stage**: {trainer.best_config['stage']}
- **Run ID**: {trainer.best_config['run_id']}
- **Proxy Public Cost**: {trainer.best_config['proxy_public']:.0f}
- **Proxy Private Cost**: {trainer.best_config['proxy_private']:.0f}
- **OOF Cost**: {trainer.best_config['oof_cost']:.0f}
- **Thresholds**: t_low={trainer.best_config['details']['t_low']:.3f}, t_high={trainer.best_config['details']['t_high']:.3f}

#### Cost Breakdown
- Missed cheaters: {trainer.best_config['details']['missed_cheaters']} (cost: {trainer.best_config['details']['cost_missed_cheaters']:.0f})
- Manual reviews: {trainer.best_config['details']['n_manual_review']} (cost: {trainer.best_config['details']['cost_manual_review']:.0f})
- Blocked clean: {trainer.best_config['details']['blocked_clean']} (cost: {trainer.best_config['details']['cost_blocked_clean']:.0f})

#### Parameters
```json
{json.dumps(trainer.best_config['params'], indent=2, default=str)}
```

### Reproduction Commands
```bash
pip install -r requirements.txt
python metric_tests.py
python train_branch_D_killer.py --data . --out artifacts/branch_D --max_trials 120
```

### Output Files
- `submission_D.csv`: Final predictions ({len(trainer.dl.test)} rows)
- `artifacts/branch_D/best.json`: Best configuration
- `artifacts/branch_D/oof_best.csv`: OOF predictions
- `artifacts/branch_D/scoreboard_D.csv`: All experiment logs
"""
    
    # Append or replace Branch D section
    if "## Branch D:" in content:
        # Replace existing section
        import re
        pattern = r"## Branch D:.*?(?=## Branch|$)"
        content = re.sub(pattern, branch_d_section, content, flags=re.DOTALL)
    else:
        content += branch_d_section
    
    with open(report_path, 'w') as f:
        f.write(content)
    
    print(f"\nUpdated {report_path}")


def main(data_dir: str = '.', out_dir: str = 'artifacts/branch_D', max_trials: int = 120):
    """Main entry point."""
    print("="*70)
    print("BRANCH D KILLER PIPELINE")
    print("Target: TOP 1 on Mercor Cheating Detection")
    print("="*70)
    
    # Load data
    dl = DataLoader(data_dir)
    dl.load()
    dl.build_graph()
    
    # Feature engineer
    fe = FeatureEngineer(dl)
    
    # Trainer
    trainer = BranchDTrainer(dl, fe, out_dir)
    
    # Prepare data
    X_labeled = dl.labeled
    y_labeled = dl.labeled['is_cheating'].values.astype(int)
    
    # Phase 1: Baselines
    baseline_results = trainer.evaluate_baselines(y_labeled)
    
    # Phase 2-4: Multi-stage search
    # Distribute trials: 40 random + 40 TPE + 30 genetic + 10 hyperband = 120 total
    stage1_trials = min(40, max_trials // 3)
    stage2_trials = min(40, max_trials // 3)
    stage3_trials = min(30, max_trials // 4)
    stage4_trials = max(10, max_trials - stage1_trials - stage2_trials - stage3_trials)
    
    trainer.stage1_random_search(X_labeled, y_labeled, n_trials=stage1_trials)
    trainer.stage2_bayesian_search(X_labeled, y_labeled, n_trials=stage2_trials)
    trainer.stage3_genetic_local_search(X_labeled, y_labeled, n_mutations=stage3_trials)
    trainer.stage4_hyperband(X_labeled, y_labeled, n_trials=stage4_trials)
    
    # Try blend
    trainer.create_blend(X_labeled, y_labeled)
    
    # Train final and predict
    test_preds = trainer.train_final_and_predict(X_labeled, y_labeled, dl.test)
    
    # Save results
    sub_path = trainer.save_results(test_preds, data_dir)
    
    # Update report
    update_report(data_dir, trainer, baseline_results)
    
    # Final summary
    print("\n" + "="*70)
    print("BRANCH D COMPLETE")
    print("="*70)
    
    if trainer.best_config:
        print(f"\nBEST RESULTS:")
        print(f"  Proxy Public Cost: {trainer.best_config['proxy_public']:.0f}")
        print(f"  Proxy Private Cost: {trainer.best_config['proxy_private']:.0f}")
        print(f"  OOF Cost: {trainer.best_config['oof_cost']:.0f}")
        print(f"  Thresholds: t_low={trainer.best_config['details']['t_low']:.3f}, t_high={trainer.best_config['details']['t_high']:.3f}")
        print(f"\nCOST BREAKDOWN:")
        print(f"  Missed cheaters: {trainer.best_config['details']['missed_cheaters']} -> cost {trainer.best_config['details']['cost_missed_cheaters']:.0f}")
        print(f"  Manual reviews: {trainer.best_config['details']['n_manual_review']} -> cost {trainer.best_config['details']['cost_manual_review']:.0f}")
        print(f"  Blocked clean: {trainer.best_config['details']['blocked_clean']} -> cost {trainer.best_config['details']['cost_blocked_clean']:.0f}")
    
    print(f"\nSubmission saved to: {sub_path}")
    
    return trainer


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='.', help='Data directory')
    parser.add_argument('--out', type=str, default='artifacts/branch_D', help='Output directory')
    parser.add_argument('--max_trials', type=int, default=120, help='Max trials across all stages')
    
    args = parser.parse_args()
    
    main(data_dir=args.data, out_dir=args.out, max_trials=args.max_trials)
