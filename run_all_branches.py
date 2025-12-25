"""
Main pipeline for Mercor Cheating Detection - 4 Branch Architecture.
Implements full end-to-end training and submission generation.

Branches:
- A: Tabular SOTA Ensemble (LightGBM, CatBoost, XGBoost)
- B: Graph Embeddings + Tabular
- C: Semi-supervised Graph Propagation (fold-safe)
- D: GNN / Pseudo-GNN approach
"""

import os
import sys
import json
import hashlib
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import networkx as nx

# Set random seeds
SEED = 42
np.random.seed(SEED)

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import our metric
from metric import score, score_with_details, get_thresholds

# Cost constants
MISS_CHEATER_COST = 500.0
BLOCK_CLEAN_COST = 100.0
MANUAL_REVIEW_COST = 50.0


class DataLoader:
    """Load and preprocess data."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.train = None
        self.test = None
        self.graph = None
        self.feature_cols = [f'feature_{i:03d}' for i in range(1, 19)]
        
    def load(self):
        """Load all data files."""
        print("Loading data...")
        self.train = pd.read_csv(self.data_dir / 'train.csv')
        self.test = pd.read_csv(self.data_dir / 'test.csv')
        self.graph_df = pd.read_csv(self.data_dir / 'social_graph.csv')
        
        with open(self.data_dir / 'feature_metadata.json') as f:
            self.feature_metadata = json.load(f)
        
        print(f"  Train: {self.train.shape}")
        print(f"  Test: {self.test.shape}")
        print(f"  Graph edges: {len(self.graph_df)}")
        
        # Split labeled and unlabeled
        self.labeled_mask = self.train['is_cheating'].notna()
        self.labeled = self.train[self.labeled_mask].copy()
        self.unlabeled = self.train[~self.labeled_mask].copy()
        
        print(f"  Labeled: {len(self.labeled)} (positives: {(self.labeled['is_cheating']==1).sum()}, negatives: {(self.labeled['is_cheating']==0).sum()})")
        print(f"  Unlabeled (high_conf_clean): {len(self.unlabeled)}")
        
        return self
    
    def build_graph(self):
        """Build NetworkX graph from edges."""
        print("Building graph...")
        self.graph = nx.Graph()
        
        # Add all nodes
        all_users = set(self.train['user_hash'].unique()) | set(self.test['user_hash'].unique())
        self.graph.add_nodes_from(all_users)
        
        # Add edges
        edges = list(zip(self.graph_df['user_a'], self.graph_df['user_b']))
        self.graph.add_edges_from(edges)
        
        print(f"  Graph nodes: {self.graph.number_of_nodes()}")
        print(f"  Graph edges: {self.graph.number_of_edges()}")
        
        return self


class FeatureEngineer:
    """Feature engineering for tabular and graph features."""
    
    def __init__(self, data_loader: DataLoader):
        self.dl = data_loader
        self.feature_cols = data_loader.feature_cols
        
    def create_tabular_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tabular features including missingness indicators and statistics."""
        features = df[self.feature_cols].copy()
        
        # Missingness indicators
        for col in self.feature_cols:
            features[f'{col}_isna'] = features[col].isna().astype(int)
        
        # Row statistics (ignoring NaN)
        features['na_count'] = features[self.feature_cols].isna().sum(axis=1)
        features['row_mean'] = features[self.feature_cols].mean(axis=1)
        features['row_std'] = features[self.feature_cols].std(axis=1)
        features['row_min'] = features[self.feature_cols].min(axis=1)
        features['row_max'] = features[self.feature_cols].max(axis=1)
        features['row_range'] = features['row_max'] - features['row_min']
        
        # Fill NaN with -999 for tree models
        features = features.fillna(-999)
        
        return features
    
    def create_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create graph-based features."""
        graph = self.dl.graph
        user_hashes = df['user_hash'].values
        
        features = pd.DataFrame(index=df.index)
        
        # Degree features
        degrees = []
        for uh in user_hashes:
            if uh in graph:
                degrees.append(graph.degree(uh))
            else:
                degrees.append(0)
        
        features['degree'] = degrees
        features['log_degree'] = np.log1p(features['degree'])
        
        # Neighbor degree statistics
        neighbor_degrees = []
        for uh in user_hashes:
            if uh in graph and graph.degree(uh) > 0:
                nd = [graph.degree(n) for n in graph.neighbors(uh)]
                neighbor_degrees.append({
                    'mean': np.mean(nd),
                    'median': np.median(nd),
                    'max': np.max(nd),
                    'min': np.min(nd)
                })
            else:
                neighbor_degrees.append({'mean': 0, 'median': 0, 'max': 0, 'min': 0})
        
        features['neighbor_degree_mean'] = [nd['mean'] for nd in neighbor_degrees]
        features['neighbor_degree_median'] = [nd['median'] for nd in neighbor_degrees]
        features['neighbor_degree_max'] = [nd['max'] for nd in neighbor_degrees]
        features['neighbor_degree_min'] = [nd['min'] for nd in neighbor_degrees]
        
        # Connected component size
        components = list(nx.connected_components(graph))
        node_to_component_size = {}
        for comp in components:
            size = len(comp)
            for node in comp:
                node_to_component_size[node] = size
        
        features['component_size'] = [node_to_component_size.get(uh, 1) for uh in user_hashes]
        features['log_component_size'] = np.log1p(features['component_size'])
        
        return features
    
    def create_fold_safe_label_features(self, df: pd.DataFrame, train_idx: np.ndarray, 
                                        train_labels: pd.Series) -> pd.DataFrame:
        """
        Create label-aware features that are fold-safe.
        Only uses labels from train_idx (not validation fold).
        """
        graph = self.dl.graph
        user_hashes = df['user_hash'].values
        
        # Create mapping from user_hash to label for train fold only
        train_users = self.dl.labeled.iloc[train_idx]['user_hash'].values
        train_y = train_labels.values
        user_to_label = dict(zip(train_users, train_y))
        
        features = pd.DataFrame(index=df.index)
        
        labeled_neighbor_counts = []
        labeled_neighbor_cheater_rates = []
        
        for uh in user_hashes:
            if uh in graph and graph.degree(uh) > 0:
                neighbors = list(graph.neighbors(uh))
                # Count labeled neighbors (from train fold only)
                labeled_neighbors = [n for n in neighbors if n in user_to_label]
                
                if len(labeled_neighbors) > 0:
                    cheater_count = sum(user_to_label[n] for n in labeled_neighbors)
                    labeled_neighbor_counts.append(len(labeled_neighbors))
                    labeled_neighbor_cheater_rates.append(cheater_count / len(labeled_neighbors))
                else:
                    labeled_neighbor_counts.append(0)
                    labeled_neighbor_cheater_rates.append(0.5)  # Prior
            else:
                labeled_neighbor_counts.append(0)
                labeled_neighbor_cheater_rates.append(0.5)  # Prior
        
        features['labeled_neighbor_count'] = labeled_neighbor_counts
        features['labeled_neighbor_cheater_rate'] = labeled_neighbor_cheater_rates
        
        return features


class ModelTrainer:
    """Train models for a specific branch."""
    
    def __init__(self, data_loader: DataLoader, feature_engineer: FeatureEngineer,
                 branch: str, out_dir: str):
        self.dl = data_loader
        self.fe = feature_engineer
        self.branch = branch
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.best_config = None
        self.best_cv_cost = float('inf')
        self.best_proxy_public_cost = float('inf')
        self.best_proxy_private_cost = float('inf')
        
    def get_proxy_split(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create deterministic stratified proxy split."""
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        public_idx, private_idx = next(skf.split(np.zeros(len(y)), y))
        return public_idx, private_idx
    
    def cv_evaluate(self, X: pd.DataFrame, y: np.ndarray, model_fn, params: dict,
                    n_folds: int = 5, use_pseudo: bool = False, pseudo_weight: float = 0.0) -> dict:
        """
        Run stratified K-fold CV and compute metrics.
        Returns OOF predictions and costs.
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        
        oof_preds = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Add pseudo-negatives if enabled
            if use_pseudo and pseudo_weight > 0:
                X_pseudo = self.fe.create_tabular_features(self.dl.unlabeled)
                y_pseudo = np.zeros(len(X_pseudo))
                sample_weights = np.concatenate([
                    np.ones(len(y_train)),
                    np.ones(len(y_pseudo)) * pseudo_weight
                ])
                X_train_full = pd.concat([X_train, X_pseudo], ignore_index=True)
                y_train_full = np.concatenate([y_train, y_pseudo])
            else:
                X_train_full = X_train
                y_train_full = y_train
                sample_weights = None
            
            # Train model
            model = model_fn(params)
            if sample_weights is not None:
                model.fit(X_train_full, y_train_full, sample_weight=sample_weights)
            else:
                model.fit(X_train_full, y_train_full)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            else:
                oof_preds[val_idx] = model.predict(X_val)
        
        # Compute costs
        cv_cost, details = score_with_details(y, oof_preds)
        
        # Proxy split evaluation
        public_idx, private_idx = self.get_proxy_split(y)
        proxy_public_cost = score(y[public_idx], oof_preds[public_idx])
        proxy_private_cost = score(y[private_idx], oof_preds[private_idx])
        
        return {
            'oof_preds': oof_preds,
            'cv_cost': cv_cost,
            'proxy_public_cost': proxy_public_cost,
            'proxy_private_cost': proxy_private_cost,
            'details': details
        }
    
    def fast_cv_evaluate(self, X: pd.DataFrame, y: np.ndarray, model_fn, params: dict) -> float:
        """Fast 2-fold CV for Stage 1 filtering."""
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        
        oof_preds = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = model_fn(params)
            model.fit(X_train, y_train)
            
            if hasattr(model, 'predict_proba'):
                oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            else:
                oof_preds[val_idx] = model.predict(X_val)
        
        return score(y, oof_preds)


class BranchA(ModelTrainer):
    """Branch A: Tabular SOTA Ensemble (LightGBM, CatBoost, XGBoost)."""
    
    def __init__(self, data_loader: DataLoader, feature_engineer: FeatureEngineer, out_dir: str):
        super().__init__(data_loader, feature_engineer, 'A', out_dir)
        
    def get_lgb_model(self, params: dict):
        """Create LightGBM model."""
        return lgb.LGBMClassifier(
            n_estimators=params.get('n_estimators', 500),
            learning_rate=params.get('learning_rate', 0.05),
            num_leaves=params.get('num_leaves', 31),
            max_depth=params.get('max_depth', -1),
            min_child_samples=params.get('min_child_samples', 20),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 0.0),
            scale_pos_weight=params.get('scale_pos_weight', 1.0),
            random_state=SEED,
            verbose=-1,
            n_jobs=-1
        )
    
    def get_cb_model(self, params: dict):
        """Create CatBoost model."""
        return cb.CatBoostClassifier(
            iterations=params.get('iterations', 500),
            learning_rate=params.get('learning_rate', 0.05),
            depth=params.get('depth', 6),
            l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
            min_data_in_leaf=params.get('min_data_in_leaf', 1),
            subsample=params.get('subsample', 0.8),
            colsample_bylevel=params.get('colsample_bylevel', 0.8),
            scale_pos_weight=params.get('scale_pos_weight', 1.0),
            random_state=SEED,
            verbose=False
        )
    
    def get_xgb_model(self, params: dict):
        """Create XGBoost model."""
        return xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 500),
            learning_rate=params.get('learning_rate', 0.05),
            max_depth=params.get('max_depth', 6),
            min_child_weight=params.get('min_child_weight', 1),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 1.0),
            scale_pos_weight=params.get('scale_pos_weight', 1.0),
            random_state=SEED,
            verbosity=0,
            use_label_encoder=False,
            n_jobs=-1
        )
    
    def stage1_hyperband(self, X: pd.DataFrame, y: np.ndarray, budget: int = 30) -> List[dict]:
        """
        Stage 1: Fast filtering with Hyperband/Successive Halving.
        Returns top K configs.
        """
        print(f"\n=== Branch A Stage 1: Hyperband (budget={budget}) ===")
        
        candidates = []
        
        def objective(trial):
            model_type = trial.suggest_categorical('model_type', ['lgb', 'cb', 'xgb'])
            
            if model_type == 'lgb':
                params = {
                    'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 300),
                    'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.2, log=True),
                    'num_leaves': trial.suggest_int('lgb_num_leaves', 15, 127),
                    'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
                    'min_child_samples': trial.suggest_int('lgb_min_child', 5, 100),
                    'subsample': trial.suggest_float('lgb_subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('lgb_colsample', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('lgb_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('lgb_lambda', 1e-8, 10.0, log=True),
                    'scale_pos_weight': trial.suggest_float('lgb_scale_pos', 1.0, 5.0),
                }
                model_fn = self.get_lgb_model
            elif model_type == 'cb':
                params = {
                    'iterations': trial.suggest_int('cb_iterations', 100, 300),
                    'learning_rate': trial.suggest_float('cb_lr', 0.01, 0.2, log=True),
                    'depth': trial.suggest_int('cb_depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('cb_l2', 1e-3, 10.0, log=True),
                    'min_data_in_leaf': trial.suggest_int('cb_min_data', 1, 100),
                    'subsample': trial.suggest_float('cb_subsample', 0.5, 1.0),
                    'colsample_bylevel': trial.suggest_float('cb_colsample', 0.5, 1.0),
                    'scale_pos_weight': trial.suggest_float('cb_scale_pos', 1.0, 5.0),
                }
                model_fn = self.get_cb_model
            else:  # xgb
                params = {
                    'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 300),
                    'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.2, log=True),
                    'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('xgb_min_child', 1, 10),
                    'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('xgb_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('xgb_lambda', 1e-8, 10.0, log=True),
                    'scale_pos_weight': trial.suggest_float('xgb_scale_pos', 1.0, 5.0),
                }
                model_fn = self.get_xgb_model
            
            cost = self.fast_cv_evaluate(X, y, model_fn, params)
            
            candidates.append({
                'model_type': model_type,
                'params': params,
                'cost': cost
            })
            
            return cost
        
        sampler = TPESampler(seed=SEED)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=budget, show_progress_bar=False)
        
        # Sort and return top 6
        candidates.sort(key=lambda x: x['cost'])
        top_k = candidates[:6]
        
        print(f"  Top 6 candidates (2-fold CV cost):")
        for i, c in enumerate(top_k):
            print(f"    {i+1}. {c['model_type']}: cost={c['cost']:.0f}")
        
        return top_k
    
    def stage2_bo(self, X: pd.DataFrame, y: np.ndarray, top_configs: List[dict], 
                  budget: int = 20) -> dict:
        """
        Stage 2: Bayesian optimization on full 5-fold CV.
        """
        print(f"\n=== Branch A Stage 2: Bayesian Optimization (budget={budget}) ===")
        
        best_result = None
        best_cost = float('inf')
        
        for config in top_configs:
            model_type = config['model_type']
            base_params = config['params']
            
            def objective(trial):
                # Fine-tune around the base params
                if model_type == 'lgb':
                    params = base_params.copy()
                    params['n_estimators'] = trial.suggest_int('n_estimators', 300, 1000)
                    params['learning_rate'] = trial.suggest_float('lr', 
                        max(0.005, base_params['learning_rate'] * 0.5),
                        min(0.3, base_params['learning_rate'] * 2), log=True)
                    model_fn = self.get_lgb_model
                elif model_type == 'cb':
                    params = base_params.copy()
                    params['iterations'] = trial.suggest_int('iterations', 300, 1000)
                    params['learning_rate'] = trial.suggest_float('lr',
                        max(0.005, base_params['learning_rate'] * 0.5),
                        min(0.3, base_params['learning_rate'] * 2), log=True)
                    model_fn = self.get_cb_model
                else:  # xgb
                    params = base_params.copy()
                    params['n_estimators'] = trial.suggest_int('n_estimators', 300, 1000)
                    params['learning_rate'] = trial.suggest_float('lr',
                        max(0.005, base_params['learning_rate'] * 0.5),
                        min(0.3, base_params['learning_rate'] * 2), log=True)
                    model_fn = self.get_xgb_model
                
                pseudo_weight = trial.suggest_float('pseudo_weight', 0.0, 0.1)
                use_pseudo = pseudo_weight > 0.001
                
                result = self.cv_evaluate(X, y, model_fn, params, 
                                         use_pseudo=use_pseudo, pseudo_weight=pseudo_weight)
                
                trial.set_user_attr('result', result)
                trial.set_user_attr('params', params)
                trial.set_user_attr('model_type', model_type)
                
                return result['proxy_public_cost']
            
            sampler = TPESampler(seed=SEED)
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(objective, n_trials=budget // len(top_configs) + 1, show_progress_bar=False)
            
            trial = study.best_trial
            result = trial.user_attrs['result']
            
            # Stability check: reject if proxy_private worsens >10%
            if self.best_proxy_private_cost < float('inf'):
                if result['proxy_private_cost'] > self.best_proxy_private_cost * 1.1:
                    print(f"    Rejected (unstable): {model_type} proxy_private={result['proxy_private_cost']:.0f}")
                    continue
            
            if result['proxy_public_cost'] < best_cost:
                best_cost = result['proxy_public_cost']
                best_result = {
                    'model_type': trial.user_attrs['model_type'],
                    'params': trial.user_attrs['params'],
                    'cv_cost': result['cv_cost'],
                    'proxy_public_cost': result['proxy_public_cost'],
                    'proxy_private_cost': result['proxy_private_cost'],
                    'details': result['details'],
                    'pseudo_weight': trial.params.get('pseudo_weight', 0.0)
                }
        
        if best_result:
            self.best_config = best_result
            self.best_cv_cost = best_result['cv_cost']
            self.best_proxy_public_cost = best_result['proxy_public_cost']
            self.best_proxy_private_cost = best_result['proxy_private_cost']
            
            print(f"\n  Best config for Branch A:")
            print(f"    Model: {best_result['model_type']}")
            print(f"    CV cost: {best_result['cv_cost']:.0f}")
            print(f"    Proxy public: {best_result['proxy_public_cost']:.0f}")
            print(f"    Proxy private: {best_result['proxy_private_cost']:.0f}")
        
        return best_result
    
    def train_final_and_predict(self, X_train: pd.DataFrame, y_train: np.ndarray,
                                X_test: pd.DataFrame) -> np.ndarray:
        """Train final model on all data and predict test."""
        if self.best_config is None:
            raise ValueError("No best config found. Run tuning first.")
        
        model_type = self.best_config['model_type']
        params = self.best_config['params']
        pseudo_weight = self.best_config.get('pseudo_weight', 0.0)
        
        # Increase iterations for final model
        if model_type == 'lgb':
            params = params.copy()
            params['n_estimators'] = min(2000, params.get('n_estimators', 500) * 2)
            model = self.get_lgb_model(params)
        elif model_type == 'cb':
            params = params.copy()
            params['iterations'] = min(2000, params.get('iterations', 500) * 2)
            model = self.get_cb_model(params)
        else:
            params = params.copy()
            params['n_estimators'] = min(2000, params.get('n_estimators', 500) * 2)
            model = self.get_xgb_model(params)
        
        # Add pseudo-negatives if enabled
        if pseudo_weight > 0.001:
            X_pseudo = self.fe.create_tabular_features(self.dl.unlabeled)
            y_pseudo = np.zeros(len(X_pseudo))
            sample_weights = np.concatenate([
                np.ones(len(y_train)),
                np.ones(len(y_pseudo)) * pseudo_weight
            ])
            X_train_full = pd.concat([X_train, X_pseudo], ignore_index=True)
            y_train_full = np.concatenate([y_train, y_pseudo])
            model.fit(X_train_full, y_train_full, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        
        # Predict
        if hasattr(model, 'predict_proba'):
            preds = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.predict(X_test)
        
        return preds
    
    def run(self, budget_fast: int = 30, budget_bo: int = 20) -> np.ndarray:
        """Run full Branch A pipeline."""
        print("\n" + "="*60)
        print("BRANCH A: Tabular SOTA Ensemble")
        print("="*60)
        
        # Prepare features
        X_labeled = self.fe.create_tabular_features(self.dl.labeled)
        y_labeled = self.dl.labeled['is_cheating'].values.astype(int)
        X_test = self.fe.create_tabular_features(self.dl.test)
        
        # Stage 1: Hyperband
        top_configs = self.stage1_hyperband(X_labeled, y_labeled, budget=budget_fast)
        
        # Stage 2: Bayesian optimization
        best = self.stage2_bo(X_labeled, y_labeled, top_configs, budget=budget_bo)
        
        # Train final and predict
        preds = self.train_final_and_predict(X_labeled, y_labeled, X_test)
        
        # Save best config
        config_path = self.out_dir / 'best.json'
        with open(config_path, 'w') as f:
            json.dump({
                'model_type': best['model_type'],
                'params': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in best['params'].items()},
                'cv_cost': float(best['cv_cost']),
                'proxy_public_cost': float(best['proxy_public_cost']),
                'proxy_private_cost': float(best['proxy_private_cost']),
                'pseudo_weight': float(best.get('pseudo_weight', 0.0))
            }, f, indent=2)
        
        return preds


class BranchB(ModelTrainer):
    """Branch B: Graph Embeddings + Tabular."""
    
    def __init__(self, data_loader: DataLoader, feature_engineer: FeatureEngineer, out_dir: str):
        super().__init__(data_loader, feature_engineer, 'B', out_dir)
        self.graph_features = None
        
    def compute_graph_features(self):
        """Compute graph features for all users."""
        print("Computing graph features...")
        
        # Get all unique users
        train_users = self.dl.train['user_hash'].unique()
        test_users = self.dl.test['user_hash'].unique()
        
        train_gf = self.fe.create_graph_features(self.dl.train)
        test_gf = self.fe.create_graph_features(self.dl.test)
        
        self.train_graph_features = train_gf
        self.test_graph_features = test_gf
        
        print(f"  Train graph features shape: {train_gf.shape}")
        print(f"  Test graph features shape: {test_gf.shape}")
    
    def get_combined_features(self, df: pd.DataFrame, graph_features: pd.DataFrame) -> pd.DataFrame:
        """Combine tabular and graph features."""
        tabular = self.fe.create_tabular_features(df)
        combined = pd.concat([tabular.reset_index(drop=True), 
                             graph_features.reset_index(drop=True)], axis=1)
        return combined
    
    def run(self, budget_fast: int = 30, budget_bo: int = 20) -> np.ndarray:
        """Run full Branch B pipeline."""
        print("\n" + "="*60)
        print("BRANCH B: Graph Embeddings + Tabular")
        print("="*60)
        
        # Compute graph features
        self.compute_graph_features()
        
        # Get labeled indices
        labeled_idx = self.dl.train[self.dl.labeled_mask].index
        
        # Prepare combined features
        X_labeled = self.get_combined_features(
            self.dl.labeled, 
            self.train_graph_features.loc[labeled_idx].reset_index(drop=True)
        )
        y_labeled = self.dl.labeled['is_cheating'].values.astype(int)
        X_test = self.get_combined_features(self.dl.test, self.test_graph_features)
        
        # Use LightGBM with graph features
        def get_model(params):
            return lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 500),
                learning_rate=params.get('learning_rate', 0.05),
                num_leaves=params.get('num_leaves', 31),
                max_depth=params.get('max_depth', -1),
                min_child_samples=params.get('min_child_samples', 20),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                reg_alpha=params.get('reg_alpha', 0.0),
                reg_lambda=params.get('reg_lambda', 0.0),
                scale_pos_weight=params.get('scale_pos_weight', 1.0),
                random_state=SEED,
                verbose=-1,
                n_jobs=-1
            )
        
        # Simple hyperparameter search
        print("\n=== Branch B: Hyperparameter Search ===")
        
        best_cost = float('inf')
        best_params = None
        
        param_space = [
            {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'scale_pos_weight': 2.0},
            {'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 8, 'scale_pos_weight': 2.5},
            {'n_estimators': 700, 'learning_rate': 0.02, 'num_leaves': 127, 'max_depth': 10, 'scale_pos_weight': 3.0},
            {'n_estimators': 300, 'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': 5, 'scale_pos_weight': 2.0},
        ]
        
        for params in param_space:
            cost = self.fast_cv_evaluate(X_labeled, y_labeled, get_model, params)
            print(f"  params={params['num_leaves']}/{params['max_depth']}, cost={cost:.0f}")
            if cost < best_cost:
                best_cost = cost
                best_params = params
        
        # Full CV on best params
        result = self.cv_evaluate(X_labeled, y_labeled, get_model, best_params)
        
        self.best_config = {
            'model_type': 'lgb_graph',
            'params': best_params,
            'cv_cost': result['cv_cost'],
            'proxy_public_cost': result['proxy_public_cost'],
            'proxy_private_cost': result['proxy_private_cost']
        }
        
        print(f"\n  Best config for Branch B:")
        print(f"    CV cost: {result['cv_cost']:.0f}")
        print(f"    Proxy public: {result['proxy_public_cost']:.0f}")
        
        # Train final model
        best_params_final = best_params.copy()
        best_params_final['n_estimators'] = min(2000, best_params['n_estimators'] * 2)
        model = get_model(best_params_final)
        model.fit(X_labeled, y_labeled)
        
        preds = model.predict_proba(X_test)[:, 1]
        
        # Save config
        config_path = self.out_dir / 'best.json'
        with open(config_path, 'w') as f:
            json.dump(self.best_config, f, indent=2)
        
        return preds


class BranchC(ModelTrainer):
    """Branch C: Semi-supervised Graph Propagation (fold-safe)."""
    
    def __init__(self, data_loader: DataLoader, feature_engineer: FeatureEngineer, out_dir: str):
        super().__init__(data_loader, feature_engineer, 'C', out_dir)
    
    def run(self, budget_fast: int = 30, budget_bo: int = 20) -> np.ndarray:
        """Run full Branch C pipeline with fold-safe label features."""
        print("\n" + "="*60)
        print("BRANCH C: Semi-supervised Graph Propagation")
        print("="*60)
        
        # Prepare base features
        X_labeled_base = self.fe.create_tabular_features(self.dl.labeled)
        y_labeled = self.dl.labeled['is_cheating'].values.astype(int)
        
        # We need to add fold-safe label features during CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        oof_preds = np.zeros(len(y_labeled))
        
        print("\n=== Branch C: Training with fold-safe label propagation ===")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled_base, y_labeled)):
            print(f"  Fold {fold+1}/5...")
            
            # Create fold-safe label features
            train_label_feats = self.fe.create_fold_safe_label_features(
                self.dl.labeled.iloc[train_idx], 
                train_idx, 
                pd.Series(y_labeled[train_idx])
            )
            val_label_feats = self.fe.create_fold_safe_label_features(
                self.dl.labeled.iloc[val_idx],
                train_idx,  # Use train fold labels only!
                pd.Series(y_labeled[train_idx])
            )
            
            # Combine features
            X_train = pd.concat([
                X_labeled_base.iloc[train_idx].reset_index(drop=True),
                train_label_feats.reset_index(drop=True)
            ], axis=1)
            X_val = pd.concat([
                X_labeled_base.iloc[val_idx].reset_index(drop=True),
                val_label_feats.reset_index(drop=True)
            ], axis=1)
            
            y_train, y_val = y_labeled[train_idx], y_labeled[val_idx]
            
            # Train model
            model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=2.5,
                random_state=SEED,
                verbose=-1,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        # Compute CV metrics
        cv_cost, details = score_with_details(y_labeled, oof_preds)
        public_idx, private_idx = self.get_proxy_split(y_labeled)
        proxy_public = score(y_labeled[public_idx], oof_preds[public_idx])
        proxy_private = score(y_labeled[private_idx], oof_preds[private_idx])
        
        print(f"\n  Branch C Results:")
        print(f"    CV cost: {cv_cost:.0f}")
        print(f"    Proxy public: {proxy_public:.0f}")
        print(f"    Proxy private: {proxy_private:.0f}")
        
        self.best_config = {
            'model_type': 'lgb_label_prop',
            'cv_cost': cv_cost,
            'proxy_public_cost': proxy_public,
            'proxy_private_cost': proxy_private
        }
        
        # Train final model on all labeled data
        # For test, create label features using all labeled data
        all_train_idx = np.arange(len(y_labeled))
        test_label_feats = self.fe.create_fold_safe_label_features(
            self.dl.test,
            all_train_idx,
            pd.Series(y_labeled)
        )
        
        X_test_base = self.fe.create_tabular_features(self.dl.test)
        X_test = pd.concat([
            X_test_base.reset_index(drop=True),
            test_label_feats.reset_index(drop=True)
        ], axis=1)
        
        # Re-create features for full training
        full_label_feats = self.fe.create_fold_safe_label_features(
            self.dl.labeled,
            all_train_idx,
            pd.Series(y_labeled)
        )
        X_labeled_full = pd.concat([
            X_labeled_base.reset_index(drop=True),
            full_label_feats.reset_index(drop=True)
        ], axis=1)
        
        final_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.5,
            random_state=SEED,
            verbose=-1,
            n_jobs=-1
        )
        final_model.fit(X_labeled_full, y_labeled)
        
        preds = final_model.predict_proba(X_test)[:, 1]
        
        # Save config
        config_path = self.out_dir / 'best.json'
        with open(config_path, 'w') as f:
            json.dump(self.best_config, f, indent=2)
        
        return preds


class BranchD(ModelTrainer):
    """Branch D: Pseudo-GNN with multi-hop aggregated features."""
    
    def __init__(self, data_loader: DataLoader, feature_engineer: FeatureEngineer, out_dir: str):
        super().__init__(data_loader, feature_engineer, 'D', out_dir)
    
    def create_multihop_features(self, df: pd.DataFrame, base_preds: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Create multi-hop aggregated features (pseudo-GNN approach)."""
        graph = self.dl.graph
        user_hashes = df['user_hash'].values
        
        features = pd.DataFrame(index=df.index)
        
        # 1-hop statistics
        hop1_stats = []
        for uh in user_hashes:
            if uh in graph and graph.degree(uh) > 0:
                neighbors = list(graph.neighbors(uh))
                hop1_stats.append({
                    'hop1_count': len(neighbors),
                    'hop1_log_count': np.log1p(len(neighbors))
                })
            else:
                hop1_stats.append({'hop1_count': 0, 'hop1_log_count': 0})
        
        features['hop1_count'] = [s['hop1_count'] for s in hop1_stats]
        features['hop1_log_count'] = [s['hop1_log_count'] for s in hop1_stats]
        
        # 2-hop statistics (unique nodes at distance 2)
        hop2_stats = []
        for uh in user_hashes:
            if uh in graph and graph.degree(uh) > 0:
                neighbors = set(graph.neighbors(uh))
                hop2_nodes = set()
                for n in neighbors:
                    hop2_nodes.update(graph.neighbors(n))
                hop2_nodes -= neighbors
                hop2_nodes.discard(uh)
                hop2_stats.append({
                    'hop2_count': len(hop2_nodes),
                    'hop2_log_count': np.log1p(len(hop2_nodes))
                })
            else:
                hop2_stats.append({'hop2_count': 0, 'hop2_log_count': 0})
        
        features['hop2_count'] = [s['hop2_count'] for s in hop2_stats]
        features['hop2_log_count'] = [s['hop2_log_count'] for s in hop2_stats]
        
        # Clustering coefficient (local)
        cc = nx.clustering(graph)
        features['clustering_coef'] = [cc.get(uh, 0) for uh in user_hashes]
        
        return features
    
    def run(self, budget_fast: int = 30, budget_bo: int = 20) -> np.ndarray:
        """Run full Branch D pipeline with pseudo-GNN features."""
        print("\n" + "="*60)
        print("BRANCH D: Pseudo-GNN (Multi-hop Features)")
        print("="*60)
        
        # Base tabular features
        X_labeled_base = self.fe.create_tabular_features(self.dl.labeled)
        y_labeled = self.dl.labeled['is_cheating'].values.astype(int)
        
        # Graph features
        labeled_idx = self.dl.train[self.dl.labeled_mask].index
        train_graph = self.fe.create_graph_features(self.dl.labeled)
        
        # Multi-hop features
        print("Computing multi-hop features...")
        train_multihop = self.create_multihop_features(self.dl.labeled)
        
        # Combine all features
        X_labeled = pd.concat([
            X_labeled_base.reset_index(drop=True),
            train_graph.reset_index(drop=True),
            train_multihop.reset_index(drop=True)
        ], axis=1)
        
        print(f"  Combined feature shape: {X_labeled.shape}")
        
        # Simple hyperparameter search
        print("\n=== Branch D: Hyperparameter Search ===")
        
        def get_model(params):
            return lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 500),
                learning_rate=params.get('learning_rate', 0.05),
                num_leaves=params.get('num_leaves', 31),
                max_depth=params.get('max_depth', 6),
                min_child_samples=params.get('min_child_samples', 20),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                scale_pos_weight=params.get('scale_pos_weight', 2.0),
                random_state=SEED,
                verbose=-1,
                n_jobs=-1
            )
        
        best_cost = float('inf')
        best_params = None
        
        param_space = [
            {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'scale_pos_weight': 2.0},
            {'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 8, 'scale_pos_weight': 2.5},
            {'n_estimators': 700, 'learning_rate': 0.02, 'num_leaves': 127, 'max_depth': 10, 'scale_pos_weight': 3.0},
        ]
        
        for params in param_space:
            cost = self.fast_cv_evaluate(X_labeled, y_labeled, get_model, params)
            print(f"  leaves={params['num_leaves']}, depth={params['max_depth']}, cost={cost:.0f}")
            if cost < best_cost:
                best_cost = cost
                best_params = params
        
        # Full CV
        result = self.cv_evaluate(X_labeled, y_labeled, get_model, best_params)
        
        print(f"\n  Branch D Results:")
        print(f"    CV cost: {result['cv_cost']:.0f}")
        print(f"    Proxy public: {result['proxy_public_cost']:.0f}")
        print(f"    Proxy private: {result['proxy_private_cost']:.0f}")
        
        self.best_config = {
            'model_type': 'lgb_multihop',
            'params': best_params,
            'cv_cost': result['cv_cost'],
            'proxy_public_cost': result['proxy_public_cost'],
            'proxy_private_cost': result['proxy_private_cost']
        }
        
        # Prepare test features
        X_test_base = self.fe.create_tabular_features(self.dl.test)
        test_graph = self.fe.create_graph_features(self.dl.test)
        test_multihop = self.create_multihop_features(self.dl.test)
        
        X_test = pd.concat([
            X_test_base.reset_index(drop=True),
            test_graph.reset_index(drop=True),
            test_multihop.reset_index(drop=True)
        ], axis=1)
        
        # Train final model
        best_params_final = best_params.copy()
        best_params_final['n_estimators'] = min(2000, best_params['n_estimators'] * 2)
        model = get_model(best_params_final)
        model.fit(X_labeled, y_labeled)
        
        preds = model.predict_proba(X_test)[:, 1]
        
        # Save config
        config_path = self.out_dir / 'best.json'
        with open(config_path, 'w') as f:
            json.dump(self.best_config, f, indent=2)
        
        return preds


def run_baselines(dl: DataLoader):
    """Run baseline predictions and compute proxy costs."""
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60)
    
    y_labeled = dl.labeled['is_cheating'].values.astype(int)
    
    # Proxy split
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    public_idx, private_idx = next(skf.split(np.zeros(len(y_labeled)), y_labeled))
    
    baselines = {
        'constant_0.01': np.full(len(y_labeled), 0.01),
        'constant_0.50': np.full(len(y_labeled), 0.50),
        'constant_0.99': np.full(len(y_labeled), 0.99),
        'uniform_random': np.random.RandomState(SEED).random(len(y_labeled))
    }
    
    print("\nBaseline costs on proxy_public split:")
    for name, preds in baselines.items():
        cost = score(y_labeled[public_idx], preds[public_idx])
        print(f"  {name}: {cost:.0f}")
    
    return baselines


def save_submission(preds: np.ndarray, test_df: pd.DataFrame, path: str):
    """Save submission file."""
    sub = pd.DataFrame({
        'user_hash': test_df['user_hash'].values,
        'prediction': preds
    })
    
    # Clip predictions to [0, 1]
    sub['prediction'] = sub['prediction'].clip(0, 1)
    
    # Check for NaN
    assert not sub['prediction'].isna().any(), "NaN predictions found!"
    
    sub.to_csv(path, index=False)
    print(f"Saved submission to {path}")
    print(f"  Rows: {len(sub)}, Prediction range: [{sub['prediction'].min():.4f}, {sub['prediction'].max():.4f}]")


def update_scoreboard(scoreboard_path: str, branch: str, stage: str, run_id: str,
                      model: str, params_json: str, cv_cost: float,
                      proxy_public: float, proxy_private: float, notes: str):
    """Update scoreboard CSV."""
    row = {
        'branch': branch,
        'stage': stage,
        'run_id': run_id,
        'model': model,
        'params_json': params_json,
        'cv_cost': cv_cost,
        'proxy_public_cost': proxy_public,
        'proxy_private_cost': proxy_private,
        'notes': notes,
        'timestamp': datetime.now().isoformat()
    }
    
    if os.path.exists(scoreboard_path):
        df = pd.read_csv(scoreboard_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    df.to_csv(scoreboard_path, index=False)


def main(data_dir: str = '.', out_dir: str = 'artifacts', 
         budget_fast: int = 30, budget_bo: int = 20):
    """Main pipeline entry point."""
    
    print("="*70)
    print("MERCOR CHEATING DETECTION - 4 BRANCH PIPELINE")
    print("="*70)
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Budget (fast): {budget_fast}, Budget (BO): {budget_bo}")
    print()
    
    # Create output directories
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize scoreboard
    scoreboard_path = out_path / 'scoreboard.csv'
    
    # Load data
    dl = DataLoader(data_dir)
    dl.load()
    dl.build_graph()
    
    # Feature engineer
    fe = FeatureEngineer(dl)
    
    # Run baselines
    run_baselines(dl)
    
    # Branch A
    branch_a = BranchA(dl, fe, out_path / 'branch_A')
    preds_a = branch_a.run(budget_fast=budget_fast, budget_bo=budget_bo)
    save_submission(preds_a, dl.test, data_dir + '/submission_A.csv')
    
    if branch_a.best_config:
        update_scoreboard(
            str(scoreboard_path), 'A', 'final', 'A_final',
            branch_a.best_config['model_type'],
            json.dumps(branch_a.best_config.get('params', {})),
            branch_a.best_config['cv_cost'],
            branch_a.best_config['proxy_public_cost'],
            branch_a.best_config['proxy_private_cost'],
            'Tabular SOTA ensemble'
        )
    
    # Branch B
    branch_b = BranchB(dl, fe, out_path / 'branch_B')
    preds_b = branch_b.run(budget_fast=budget_fast, budget_bo=budget_bo)
    save_submission(preds_b, dl.test, data_dir + '/submission_B.csv')
    
    if branch_b.best_config:
        update_scoreboard(
            str(scoreboard_path), 'B', 'final', 'B_final',
            branch_b.best_config['model_type'],
            json.dumps(branch_b.best_config.get('params', {})),
            branch_b.best_config['cv_cost'],
            branch_b.best_config['proxy_public_cost'],
            branch_b.best_config['proxy_private_cost'],
            'Graph embeddings + tabular'
        )
    
    # Branch C
    branch_c = BranchC(dl, fe, out_path / 'branch_C')
    preds_c = branch_c.run(budget_fast=budget_fast, budget_bo=budget_bo)
    save_submission(preds_c, dl.test, data_dir + '/submission_C.csv')
    
    if branch_c.best_config:
        update_scoreboard(
            str(scoreboard_path), 'C', 'final', 'C_final',
            branch_c.best_config['model_type'],
            json.dumps(branch_c.best_config.get('params', {})),
            branch_c.best_config['cv_cost'],
            branch_c.best_config['proxy_public_cost'],
            branch_c.best_config['proxy_private_cost'],
            'Semi-supervised label propagation'
        )
    
    # Branch D
    branch_d = BranchD(dl, fe, out_path / 'branch_D')
    preds_d = branch_d.run(budget_fast=budget_fast, budget_bo=budget_bo)
    save_submission(preds_d, dl.test, data_dir + '/submission_D.csv')
    
    if branch_d.best_config:
        update_scoreboard(
            str(scoreboard_path), 'D', 'final', 'D_final',
            branch_d.best_config['model_type'],
            json.dumps(branch_d.best_config.get('params', {})),
            branch_d.best_config['cv_cost'],
            branch_d.best_config['proxy_public_cost'],
            branch_d.best_config['proxy_private_cost'],
            'Pseudo-GNN multi-hop'
        )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\nGenerated submissions:")
    for branch in ['A', 'B', 'C', 'D']:
        sub_path = f"{data_dir}/submission_{branch}.csv"
        if os.path.exists(sub_path):
            df = pd.read_csv(sub_path)
            print(f"  submission_{branch}.csv: {len(df)} rows")
    
    print(f"\nScoreboard saved to: {scoreboard_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='.', help='Data directory')
    parser.add_argument('--out', type=str, default='artifacts', help='Output directory')
    parser.add_argument('--budget_fast', type=int, default=30, help='Stage 1 budget')
    parser.add_argument('--budget_bo', type=int, default=20, help='Stage 2 budget')
    
    args = parser.parse_args()
    
    main(data_dir=args.data, out_dir=args.out, 
         budget_fast=args.budget_fast, budget_bo=args.budget_bo)
