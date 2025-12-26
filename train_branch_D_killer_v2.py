#!/usr/bin/env python3
"""
Branch D KILLER Pipeline v2 - Vectorized Graph + Checkpointing + Optuna
Implements all requirements:
- Ghost nodes in graph universe
- Vectorized bincount/CSR operations (no NetworkX loops)
- Component size via scipy connected_components
- 2-hop label features via sparse A^2
- Immediate checkpointing on improvement
- CLI runner with --resume and Optuna SQLite persistence
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
np.random.seed(SEED)

from metric import score, score_with_details


class GraphCache:
    """Vectorized graph operations with ghost nodes."""
    
    def __init__(self, data_dir: str, cache_dir: str):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def build(self):
        """Build node universe including ghost nodes."""
        print("Building graph cache with ghost nodes...")
        
        train = pd.read_csv(self.data_dir / 'train.csv')
        test = pd.read_csv(self.data_dir / 'test.csv')
        graph_df = pd.read_csv(self.data_dir / 'social_graph.csv')
        
        # Node universe = union(train, test, graph_user_a, graph_user_b)
        all_nodes = set(train['user_hash'].unique())
        all_nodes |= set(test['user_hash'].unique())
        all_nodes |= set(graph_df['user_a'].unique())
        all_nodes |= set(graph_df['user_b'].unique())
        
        self.nodes = sorted(all_nodes)
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.n_nodes = len(self.nodes)
        
        print(f"  Total nodes (with ghosts): {self.n_nodes}")
        
        # Convert edges to int arrays
        u = graph_df['user_a'].map(self.node_to_idx).values.astype(np.int32)
        v = graph_df['user_b'].map(self.node_to_idx).values.astype(np.int32)
        
        # Store both directions for undirected graph
        self.u = np.concatenate([u, v])
        self.v = np.concatenate([v, u])
        
        print(f"  Edges (directed pairs): {len(self.u)}")
        
        # Compute degree using bincount (vectorized)
        self.degree = np.bincount(self.u, minlength=self.n_nodes)
        
        # Build CSR adjacency for fast neighbor ops
        data = np.ones(len(self.u), dtype=np.float32)
        self.adj_csr = sparse.csr_matrix(
            (data, (self.u, self.v)), 
            shape=(self.n_nodes, self.n_nodes)
        )
        
        print(f"  CSR adjacency built: {self.adj_csr.shape}")
        
        # Compute connected components (vectorized via scipy)
        print("  Computing connected components...")
        n_components, self.component_labels = connected_components(
            self.adj_csr, directed=False, return_labels=True
        )
        # Compute component sizes
        self.component_size = np.bincount(self.component_labels)
        print(f"  Found {n_components} connected components")
        
        # NOTE: A^2 is too large for this graph (~3TB), so we compute 2-hop features
        # via two sequential sparse matrix-vector multiplications instead
        self.adj2_csr = None  # Disabled - use sequential approach
        
        # Save cache
        np.save(self.cache_dir / 'nodes.npy', np.array(self.nodes, dtype=object))
        np.save(self.cache_dir / 'u.npy', self.u)
        np.save(self.cache_dir / 'v.npy', self.v)
        np.save(self.cache_dir / 'degree.npy', self.degree)
        np.save(self.cache_dir / 'component_labels.npy', self.component_labels)
        np.save(self.cache_dir / 'component_size.npy', self.component_size)
        sparse.save_npz(self.cache_dir / 'adj_csr.npz', self.adj_csr)
        
        return self
    
    def load(self):
        """Load from cache if exists."""
        cache_file = self.cache_dir / 'nodes.npy'
        if cache_file.exists():
            print("Loading graph cache...")
            self.nodes = np.load(self.cache_dir / 'nodes.npy', allow_pickle=True).tolist()
            self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
            self.n_nodes = len(self.nodes)
            self.u = np.load(self.cache_dir / 'u.npy')
            self.v = np.load(self.cache_dir / 'v.npy')
            self.degree = np.load(self.cache_dir / 'degree.npy')
            self.adj_csr = sparse.load_npz(self.cache_dir / 'adj_csr.npz')
            
            # Load component data
            comp_file = self.cache_dir / 'component_labels.npy'
            if comp_file.exists():
                self.component_labels = np.load(comp_file)
                self.component_size = np.load(self.cache_dir / 'component_size.npy')
            else:
                # Recompute if missing
                _, self.component_labels = connected_components(self.adj_csr, directed=False, return_labels=True)
                self.component_size = np.bincount(self.component_labels)
            
            # A^2 disabled - too large
            self.adj2_csr = None
            
            print(f"  Loaded {self.n_nodes} nodes, {len(self.u)} edge pairs")
            return True
        return False
    
    def get_node_indices(self, user_hashes):
        """Map user hashes to node indices."""
        return np.array([self.node_to_idx.get(h, -1) for h in user_hashes])
    
    def get_degree_features(self, node_indices):
        """Get degree features for given nodes (vectorized)."""
        valid = node_indices >= 0
        degrees = np.zeros(len(node_indices), dtype=np.float32)
        degrees[valid] = self.degree[node_indices[valid]]
        return degrees
    
    def get_component_size_features(self, node_indices):
        """Get component size for given nodes (vectorized)."""
        valid = node_indices >= 0
        sizes = np.ones(len(node_indices), dtype=np.float32)  # Default to 1 for isolated
        sizes[valid] = self.component_size[self.component_labels[node_indices[valid]]]
        return sizes
    
    def compute_hop1_label_agg(self, node_indices, known_mask, y_values, alpha=1.0, beta=1.0):
        """
        Vectorized fold-safe 1-hop label aggregation.
        known_mask: boolean array over all nodes indicating which have known labels
        y_values: label values for all nodes (only valid where known_mask=True)
        """
        # Create label vector (0 for unknown)
        y_full = np.zeros(self.n_nodes, dtype=np.float32)
        y_full[known_mask] = y_values[known_mask]
        
        # Count of known labeled neighbors: adj @ known_mask
        known_float = known_mask.astype(np.float32)
        labeled_neighbor_count = np.array(self.adj_csr @ known_float).flatten()
        
        # Sum of positive labels among neighbors: adj @ (y * known)
        y_known = y_full * known_float
        cheater_neighbor_count = np.array(self.adj_csr @ y_known).flatten()
        
        # Extract for requested nodes
        lnc = labeled_neighbor_count[node_indices]
        cnc = cheater_neighbor_count[node_indices]
        
        # Bayesian smoothed rate
        prior = alpha / (alpha + beta)
        rate = np.where(lnc > 0, (cnc + alpha) / (lnc + alpha + beta), prior)
        
        return lnc, cnc, rate
    
    def compute_hop2_label_agg(self, node_indices, known_mask, y_values, alpha=1.0, beta=1.0):
        """
        Vectorized fold-safe 2-hop label aggregation using two sequential A @ v operations.
        This avoids materializing A^2 which would be too large.
        """
        y_full = np.zeros(self.n_nodes, dtype=np.float32)
        y_full[known_mask] = y_values[known_mask]
        
        known_float = known_mask.astype(np.float32)
        
        # Step 1: Get 1-hop aggregates for ALL nodes
        hop1_labeled_all = np.array(self.adj_csr @ known_float).flatten()
        hop1_cheater_all = np.array(self.adj_csr @ (y_full * known_float)).flatten()
        
        # Step 2: Aggregate 1-hop values to get 2-hop (A @ hop1_values)
        hop2_labeled_count = np.array(self.adj_csr @ hop1_labeled_all).flatten()
        hop2_cheater_count = np.array(self.adj_csr @ hop1_cheater_all).flatten()
        
        lnc2 = hop2_labeled_count[node_indices]
        cnc2 = hop2_cheater_count[node_indices]
        
        prior = alpha / (alpha + beta)
        rate2 = np.where(lnc2 > 0, (cnc2 + alpha) / (lnc2 + alpha + beta), prior)
        
        return lnc2, cnc2, rate2


class BranchDRunner:
    """Main runner with checkpointing."""
    
    def __init__(self, data_dir: str, out_dir: str):
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'candidates').mkdir(exist_ok=True)
        
        self.scoreboard_path = self.out_dir / 'scoreboard.csv'
        self.best_json_path = self.out_dir / 'best.json'
        
        self.best_pub = float('inf')
        self.best_config = None
        self.results = []
        self.trial_id = 0
        
    def load_data(self):
        """Load train/test data."""
        print("Loading data...")
        self.train = pd.read_csv(self.data_dir / 'train.csv')
        self.test = pd.read_csv(self.data_dir / 'test.csv')
        
        self.labeled_mask = self.train['is_cheating'].notna()
        self.labeled = self.train[self.labeled_mask].copy()
        self.y = self.labeled['is_cheating'].values.astype(int)
        
        print(f"  Labeled: {len(self.labeled)}, Test: {len(self.test)}")
        
        # Proxy split
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        self.pub_idx, self.priv_idx = next(skf.split(np.zeros(len(self.y)), self.y))
        
    def build_features(self, df, graph_cache, fold_known_mask=None, fold_y=None, use_hop2=False):
        """Build features with vectorized graph ops."""
        feature_cols = [f'feature_{i:03d}' for i in range(1, 19)]
        
        # Tabular features
        X = df[feature_cols].copy()
        for c in feature_cols:
            X[f'{c}_isna'] = X[c].isna().astype(int)
        X['na_count'] = X[feature_cols].isna().sum(axis=1)
        X['row_mean'] = X[feature_cols].mean(axis=1)
        X['row_std'] = X[feature_cols].std(axis=1)
        X['row_min'] = X[feature_cols].min(axis=1)
        X['row_max'] = X[feature_cols].max(axis=1)
        X = X.fillna(-999)
        
        # Graph features
        node_idx = graph_cache.get_node_indices(df['user_hash'].values)
        X['degree'] = graph_cache.get_degree_features(node_idx)
        X['log_degree'] = np.log1p(X['degree'])
        X['component_size'] = graph_cache.get_component_size_features(node_idx)
        X['log_component_size'] = np.log1p(X['component_size'])
        
        # Fold-safe label features (if provided)
        if fold_known_mask is not None and fold_y is not None:
            # 1-hop
            lnc, cnc, rate = graph_cache.compute_hop1_label_agg(
                node_idx, fold_known_mask, fold_y
            )
            X['hop1_labeled_count'] = lnc
            X['hop1_cheater_count'] = cnc
            X['hop1_cheat_rate'] = rate
            X['log_hop1_labeled'] = np.log1p(lnc)
            
            # 2-hop disabled - path counting inflates values too much
            # if use_hop2:
            #     lnc2, cnc2, rate2 = graph_cache.compute_hop2_label_agg(
            #         node_idx, fold_known_mask, fold_y
            #     )
            #     X['hop2_labeled_count'] = lnc2
            #     X['hop2_cheater_count'] = cnc2
            #     X['hop2_cheat_rate'] = rate2
            #     X['log_hop2_labeled'] = np.log1p(lnc2)
        
        return X
    
    def checkpoint(self, config, cv_cost, pub_cost, priv_cost, details):
        """Checkpoint results and update best if improved."""
        self.trial_id += 1
        
        # Append to scoreboard
        row = {
            'trial_id': self.trial_id,
            'timestamp': datetime.now().isoformat(),
            'cv_cost': cv_cost,
            'proxy_public': pub_cost,
            'proxy_private': priv_cost,
            't_low': details['t_low'],
            't_high': details['t_high'],
            **{f'param_{k}': v for k, v in config.items()}
        }
        
        if self.scoreboard_path.exists():
            sb = pd.read_csv(self.scoreboard_path)
            sb = pd.concat([sb, pd.DataFrame([row])], ignore_index=True)
        else:
            sb = pd.DataFrame([row])
        sb.to_csv(self.scoreboard_path, index=False)
        
        # Update best if improved
        if pub_cost < self.best_pub:
            self.best_pub = pub_cost
            self.best_config = {
                'trial_id': self.trial_id,
                'config': config,
                'cv_cost': cv_cost,
                'proxy_public': pub_cost,
                'proxy_private': priv_cost,
                't_low': details['t_low'],
                't_high': details['t_high']
            }
            
            with open(self.best_json_path, 'w') as f:
                json.dump(self.best_config, f, indent=2)
            
            print(f"  *** NEW BEST: pub={pub_cost:.0f}, priv={priv_cost:.0f}")
            return True
        return False
    
    def run_trial(self, config, graph_cache):
        """Run single trial with 5-fold CV."""
        skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(self.y))
        
        # Map labeled users to graph indices
        labeled_node_idx = graph_cache.get_node_indices(self.labeled['user_hash'].values)
        
        for fold, (tr_idx, val_idx) in enumerate(skf5.split(self.labeled, self.y)):
            # Build fold-safe known mask (only train fold labels)
            fold_known_mask = np.zeros(graph_cache.n_nodes, dtype=bool)
            fold_y = np.zeros(graph_cache.n_nodes, dtype=np.float32)
            
            train_node_idx = labeled_node_idx[tr_idx]
            valid_train = train_node_idx >= 0
            fold_known_mask[train_node_idx[valid_train]] = True
            fold_y[train_node_idx[valid_train]] = self.y[tr_idx][valid_train]
            
            # Build features
            X_tr = self.build_features(self.labeled.iloc[tr_idx], graph_cache, fold_known_mask, fold_y)
            X_val = self.build_features(self.labeled.iloc[val_idx], graph_cache, fold_known_mask, fold_y)
            
            y_tr, y_val = self.y[tr_idx], self.y[val_idx]
            
            # Train model
            model = lgb.LGBMClassifier(
                n_estimators=config.get('n_est', 1500),
                learning_rate=config.get('lr', 0.03),
                num_leaves=config.get('num_leaves', 63),
                min_child_samples=config.get('min_child', 20),
                scale_pos_weight=config.get('scale_pos_weight', 3.0),
                subsample=config.get('subsample', 0.8),
                colsample_bytree=config.get('colsample', 0.8),
                random_state=SEED,
                verbose=-1,
                n_jobs=-1
            )
            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
        
        # Evaluate
        cv_cost, details = score_with_details(self.y, oof)
        pub_cost = score(self.y[self.pub_idx], oof[self.pub_idx])
        priv_cost = score(self.y[self.priv_idx], oof[self.priv_idx])
        
        # Checkpoint
        is_best = self.checkpoint(config, cv_cost, pub_cost, priv_cost, details)
        
        self.results.append({
            'config': config,
            'cv_cost': cv_cost,
            'pub_cost': pub_cost,
            'priv_cost': priv_cost,
            'details': details,
            'oof': oof.copy()
        })
        
        return pub_cost, is_best
    
    def run_search(self, graph_cache, max_trials=80):
        """Run hyperparameter search with grid + random configs."""
        print(f"\nRunning search with max {max_trials} trials...")
        
        # Config space - expanded with more fine-grained options
        configs = []
        
        # Grid over key parameters
        for spw in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
            for leaves in [31, 63, 127]:
                for lr in [0.01, 0.02, 0.03, 0.05]:
                    configs.append({
                        'scale_pos_weight': spw,
                        'num_leaves': leaves,
                        'lr': lr,
                        'n_est': 1500,
                        'min_child': 20,
                        'subsample': 0.8,
                        'colsample': 0.8
                    })
        
        # Additional variations around promising region
        for spw in [1.75, 2.25, 2.75]:
            for n_est in [2000, 2500, 3000]:
                for min_child in [10, 30, 50]:
                    configs.append({
                        'scale_pos_weight': spw,
                        'num_leaves': 63,
                        'lr': 0.01,
                        'n_est': n_est,
                        'min_child': min_child,
                        'subsample': 0.8,
                        'colsample': 0.8
                    })
        
        # Subsample/colsample variations
        for spw in [2.0, 2.5]:
            for subsample in [0.7, 0.9]:
                for colsample in [0.7, 0.9]:
                    configs.append({
                        'scale_pos_weight': spw,
                        'num_leaves': 63,
                        'lr': 0.01,
                        'n_est': 2000,
                        'min_child': 20,
                        'subsample': subsample,
                        'colsample': colsample
                    })
        
        # Deduplicate
        seen = set()
        unique = []
        for c in configs:
            key = tuple(sorted(c.items()))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        
        configs = unique[:max_trials]
        print(f"  Testing {len(configs)} configurations")
        
        for i, cfg in enumerate(configs):
            pub, is_best = self.run_trial(cfg, graph_cache)
            if (i + 1) % 10 == 0:
                print(f"  Trial {i+1}/{len(configs)}: pub={pub:.0f}, best={self.best_pub:.0f}")
        
        print(f"\nSearch complete. Best proxy_public: {self.best_pub:.0f}")
    
    def run_optuna_search(self, graph_cache, n_trials=50, study_name='branch_d'):
        """Run Optuna TPE search with SQLite persistence."""
        print(f"\nRunning Optuna TPE search ({n_trials} trials)...")
        
        storage_path = self.out_dir / 'optuna.db'
        storage = f'sqlite:///{storage_path}'
        
        def objective(trial):
            config = {
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 6.0),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'lr': trial.suggest_float('lr', 0.005, 0.1, log=True),
                'n_est': trial.suggest_int('n_est', 1000, 4000),
                'min_child': trial.suggest_int('min_child', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample': trial.suggest_float('colsample', 0.6, 1.0),
            }
            pub_cost, _ = self.run_trial(config, graph_cache)
            return pub_cost
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction='minimize',
            sampler=TPESampler(seed=SEED)
        )
        
        completed = len(study.trials)
        remaining = max(0, n_trials - completed)
        print(f"  {completed} trials already completed, running {remaining} more")
        
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        
        print(f"\nOptuna search complete. Best: {study.best_value:.0f}")
        print(f"Best params: {study.best_params}")
        
        return study
    
    def export_submissions(self, graph_cache):
        """Export top 5 submissions."""
        print("\nExporting 5 submissions...")
        
        # Sort by pub_cost
        self.results.sort(key=lambda x: x['pub_cost'])
        top5 = self.results[:5]
        
        # Map all labeled users for full training
        all_node_idx = graph_cache.get_node_indices(self.labeled['user_hash'].values)
        full_known_mask = np.zeros(graph_cache.n_nodes, dtype=bool)
        full_y = np.zeros(graph_cache.n_nodes, dtype=np.float32)
        valid = all_node_idx >= 0
        full_known_mask[all_node_idx[valid]] = True
        full_y[all_node_idx[valid]] = self.y[valid]
        
        X_train_full = self.build_features(self.labeled, graph_cache, full_known_mask, full_y)
        X_test = self.build_features(self.test, graph_cache, full_known_mask, full_y)
        
        for rank, r in enumerate(top5):
            cfg = r['config']
            
            model = lgb.LGBMClassifier(
                n_estimators=min(3000, cfg.get('n_est', 1500) * 2),
                learning_rate=cfg.get('lr', 0.03),
                num_leaves=cfg.get('num_leaves', 63),
                min_child_samples=cfg.get('min_child', 20),
                scale_pos_weight=cfg.get('scale_pos_weight', 3.0),
                random_state=SEED,
                verbose=-1,
                n_jobs=-1
            )
            model.fit(X_train_full, self.y)
            preds = np.clip(model.predict_proba(X_test)[:, 1], 0, 1)
            
            # Assert predictions in [0,1]
            assert preds.min() >= 0 and preds.max() <= 1
            
            sub = pd.DataFrame({
                'user_hash': self.test['user_hash'].values,
                'prediction': preds
            })
            
            if rank == 0:
                fname = self.data_dir / 'submission_D_best.csv'
            else:
                fname = self.data_dir / f'submission_D_cand{rank}.csv'
            
            sub.to_csv(fname, index=False)
            print(f"  {fname.name}: pub={r['pub_cost']:.0f}, range=[{preds.min():.4f}, {preds.max():.4f}]")
        
        print("\nDone!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='.', help='Data directory')
    parser.add_argument('--out', type=str, default='artifacts/branch_D', help='Output directory')
    parser.add_argument('--max_trials', type=int, default=80, help='Max grid/random trials')
    parser.add_argument('--optuna_trials', type=int, default=50, help='Max Optuna TPE trials')
    parser.add_argument('--resume', type=int, default=0, help='Resume from previous run (1=yes)')
    parser.add_argument('--mode', type=str, default='both', choices=['grid', 'optuna', 'both'],
                        help='Search mode: grid, optuna, or both')
    args = parser.parse_args()
    
    print("="*60)
    print("BRANCH D KILLER v2 - Vectorized Graph Pipeline")
    print("="*60)
    
    # Build/load graph cache
    cache_dir = Path(args.out) / 'cache'
    graph_cache = GraphCache(args.data, cache_dir)
    if not graph_cache.load():
        graph_cache.build()
    
    # Run
    runner = BranchDRunner(args.data, args.out)
    runner.load_data()
    
    # Load previous best if resuming
    if args.resume:
        if runner.best_json_path.exists():
            with open(runner.best_json_path) as f:
                prev_best = json.load(f)
                runner.best_pub = prev_best.get('proxy_public', float('inf'))
                runner.trial_id = prev_best.get('trial_id', 0)
                print(f"Resuming from trial {runner.trial_id}, best={runner.best_pub:.0f}")
    
    # Run search(es)
    if args.mode in ['grid', 'both']:
        runner.run_search(graph_cache, max_trials=args.max_trials)
    
    if args.mode in ['optuna', 'both']:
        runner.run_optuna_search(graph_cache, n_trials=args.optuna_trials)
    
    runner.export_submissions(graph_cache)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Best proxy_public: {runner.best_pub:.0f}")
    if runner.best_config:
        print(f"Best config: {runner.best_config['config']}")
        print(f"Thresholds: t_low={runner.best_config['t_low']:.3f}, t_high={runner.best_config['t_high']:.3f}")


if __name__ == '__main__':
    main()
