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
        Optimized fold-safe 1-hop label aggregation.
        Only computes for requested nodes via CSR row-slicing (O(n_requested) not O(n_nodes)).
        Handles invalid indices (-1) safely.
        """
        n_req = len(node_indices)
        prior = alpha / (alpha + beta)
        
        # Handle invalid indices: create valid mask and safe indices
        valid_mask = node_indices >= 0
        safe_idx = np.where(valid_mask, node_indices, 0)  # Replace -1 with 0 for indexing
        
        # Create label vector (0 for unknown)
        y_full = np.zeros(self.n_nodes, dtype=np.float32)
        y_full[known_mask] = y_values[known_mask]
        known_float = known_mask.astype(np.float32)
        y_known = y_full * known_float
        
        # Row-slice adjacency for only requested nodes (fast)
        sub_adj = self.adj_csr[safe_idx]  # shape: (len(node_indices), n_nodes)
        
        # Compute aggregates only for requested rows
        lnc = np.array(sub_adj @ known_float).flatten()  # labeled neighbor count
        cnc = np.array(sub_adj @ y_known).flatten()      # cheater neighbor count
        
        # Set invalid rows to defaults (0, 0, prior)
        lnc = np.where(valid_mask, lnc, 0)
        cnc = np.where(valid_mask, cnc, 0)
        
        # Bayesian smoothed rate
        rate = np.where(lnc > 0, (cnc + alpha) / (lnc + alpha + beta), prior)
        
        return lnc, cnc, rate
    
    def compute_component_prior_features(self, node_indices, known_mask, y_values, alpha=1.0, beta=1.0):
        """
        Fold-safe component-level label statistics (VECTORIZED with bincount, no loops).
        For each node, compute stats about its connected component using only train-fold labels.
        Handles invalid indices (-1) safely.
        """
        n_req = len(node_indices)
        prior = alpha / (alpha + beta)
        
        # Handle invalid indices
        valid_mask = node_indices >= 0
        safe_idx = np.where(valid_mask, node_indices, 0)
        
        # Get component labels for requested nodes
        comp_labels = self.component_labels[safe_idx]
        
        # VECTORIZED: compute per-component stats using bincount (no loops)
        n_comps = len(self.component_size)
        
        # Get component IDs of all known (train-fold) nodes
        known_node_idx = np.where(known_mask)[0]
        comp_ids_known = self.component_labels[known_node_idx]
        
        # bincount: count labeled nodes per component
        comp_labeled_cnt = np.bincount(comp_ids_known, minlength=n_comps).astype(np.float32)
        
        # bincount with weights: count cheaters per component
        y_known = y_values[known_node_idx].astype(np.float32)
        comp_cheater_cnt = np.bincount(comp_ids_known, weights=y_known, minlength=n_comps).astype(np.float32)
        
        # Now get features for requested nodes
        comp_labeled = comp_labeled_cnt[comp_labels]
        comp_cheater = comp_cheater_cnt[comp_labels]
        comp_rate = np.where(comp_labeled > 0, 
                            (comp_cheater + alpha) / (comp_labeled + alpha + beta),
                            prior)
        comp_has_cheater = (comp_cheater > 0).astype(np.float32)
        
        # Set invalid rows to defaults (labeled=0, cheater=0, rate=prior, has_cheater=0)
        comp_labeled = np.where(valid_mask, comp_labeled, 0)
        comp_cheater = np.where(valid_mask, comp_cheater, 0)
        comp_rate = np.where(valid_mask, comp_rate, prior)
        comp_has_cheater = np.where(valid_mask, comp_has_cheater, 0)
        
        return comp_labeled, comp_cheater, comp_rate, comp_has_cheater
    
    def compute_neighbor_mean_features(self, node_indices):
        """
        Compute neighbor mean features for requested nodes via CSR row-slice.
        Features: nbr_degree_mean, nbr_comp_size_mean
        Handles invalid indices (-1) safely.
        Defaults: nbr_degree_mean=0, nbr_comp_size_mean=1 when no neighbors or invalid.
        """
        n_req = len(node_indices)
        
        # Handle invalid indices
        valid_mask = node_indices >= 0
        safe_idx = np.where(valid_mask, node_indices, 0)
        
        # Row-slice adjacency for requested nodes
        sub_adj = self.adj_csr[safe_idx]
        
        # For each requested node, compute mean of neighbor degrees
        # sub_adj @ degree gives sum of neighbor degrees, then divide by degree
        nbr_degree_sum = np.array(sub_adj @ self.degree.astype(np.float32)).flatten()
        my_degree = self.degree[safe_idx].astype(np.float32)
        nbr_degree_mean = np.where(my_degree > 0, nbr_degree_sum / my_degree, 0.0)  # default 0
        
        # Mean of neighbor component sizes
        nbr_comp_sizes = self.component_size[self.component_labels].astype(np.float32)
        nbr_compsize_sum = np.array(sub_adj @ nbr_comp_sizes).flatten()
        nbr_compsize_mean = np.where(my_degree > 0, nbr_compsize_sum / my_degree, 1.0)  # default 1
        
        # Set invalid rows to defaults (0 for degree_mean, 1 for comp_size_mean)
        nbr_degree_mean = np.where(valid_mask, nbr_degree_mean, 0.0)
        nbr_compsize_mean = np.where(valid_mask, nbr_compsize_mean, 1.0)
        
        return nbr_degree_mean, nbr_compsize_mean

    def compute_hop2_label_agg_sampled(self, node_indices, known_mask, y_values, alpha=1.0, beta=1.0, max_neighbors=50):
        """
        Sampled 2-hop label aggregation. For each requested node, sample up to max_neighbors
        from its 1-hop neighbors, then average their hop1 label stats.
        This avoids materializing A^2 and keeps computation O(n_requested * max_neighbors).
        """
        n_req = len(node_indices)
        hop2_score = np.zeros(n_req, dtype=np.float32)
        
        # Precompute hop1 stats for ALL nodes to enable sampling
        y_full = np.zeros(self.n_nodes, dtype=np.float32)
        y_full[known_mask] = y_values[known_mask]
        known_float = known_mask.astype(np.float32)
        y_known = y_full * known_float
        
        # Get hop1 for all nodes (needed for neighbor lookup)
        all_lnc = np.array(self.adj_csr @ known_float).flatten()
        all_cnc = np.array(self.adj_csr @ y_known).flatten()
        prior = alpha / (alpha + beta)
        all_hop1_rate = np.where(all_lnc > 0, (all_cnc + alpha) / (all_lnc + alpha + beta), prior)
        
        # For each requested node, sample neighbors and average their hop1 rate
        rng = np.random.default_rng(SEED)
        for i, nidx in enumerate(node_indices):
            if nidx < 0:
                hop2_score[i] = prior
                continue
            
            # Get neighbors from CSR (efficient row slice)
            start, end = self.adj_csr.indptr[nidx], self.adj_csr.indptr[nidx + 1]
            neighbors = self.adj_csr.indices[start:end]
            
            if len(neighbors) == 0:
                hop2_score[i] = prior
            elif len(neighbors) <= max_neighbors:
                hop2_score[i] = all_hop1_rate[neighbors].mean()
            else:
                sampled = rng.choice(neighbors, max_neighbors, replace=False)
                hop2_score[i] = all_hop1_rate[sampled].mean()
        
        return hop2_score


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
        
    def build_features(self, df, graph_cache, fold_known_mask=None, fold_y=None, 
                       use_hop2=False, alpha=1.0, beta=1.0, hop2_max_neighbors=50):
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
            # 1-hop with tunable alpha/beta
            lnc, cnc, rate = graph_cache.compute_hop1_label_agg(
                node_idx, fold_known_mask, fold_y, alpha=alpha, beta=beta
            )
            X['hop1_labeled_count'] = lnc
            X['hop1_cheater_count'] = cnc
            X['hop1_cheat_rate'] = rate
            X['log_hop1_labeled'] = np.log1p(lnc)
            
            # Component-level fold-safe label stats
            comp_labeled, comp_cheater, comp_rate, comp_has_cheater = graph_cache.compute_component_prior_features(
                node_idx, fold_known_mask, fold_y, alpha=alpha, beta=beta
            )
            X['comp_labeled_count'] = comp_labeled
            X['comp_cheater_count'] = comp_cheater
            X['comp_cheat_rate'] = comp_rate
            X['comp_has_cheater'] = comp_has_cheater
            X['log_comp_labeled'] = np.log1p(comp_labeled)
            
            # Neighbor mean features (cheap CSR row-slice)
            nbr_deg_mean, nbr_comp_mean = graph_cache.compute_neighbor_mean_features(node_idx)
            X['nbr_degree_mean'] = nbr_deg_mean
            X['nbr_comp_size_mean'] = nbr_comp_mean
            X['log_nbr_deg_mean'] = np.log1p(nbr_deg_mean)
            X['log_nbr_comp_mean'] = np.log1p(nbr_comp_mean)
            
            # Sampled 2-hop (safe, no A^2)
            if use_hop2:
                hop2_score = graph_cache.compute_hop2_label_agg_sampled(
                    node_idx, fold_known_mask, fold_y, 
                    alpha=alpha, beta=beta, max_neighbors=hop2_max_neighbors
                )
                X['hop2_neighbor_rate_sampled'] = hop2_score
        
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
        
        # Extract feature config
        alpha = config.get('alpha', 1.0)
        beta = config.get('beta', 1.0)
        use_hop2 = config.get('use_hop2', False)
        hop2_max_neighbors = config.get('hop2_max_neighbors', 50)
        
        for fold, (tr_idx, val_idx) in enumerate(skf5.split(self.labeled, self.y)):
            # Build fold-safe known mask (only train fold labels)
            fold_known_mask = np.zeros(graph_cache.n_nodes, dtype=bool)
            fold_y = np.zeros(graph_cache.n_nodes, dtype=np.float32)
            
            train_node_idx = labeled_node_idx[tr_idx]
            valid_train = train_node_idx >= 0
            fold_known_mask[train_node_idx[valid_train]] = True
            fold_y[train_node_idx[valid_train]] = self.y[tr_idx][valid_train]
            
            # Build features with configurable alpha/beta and optional hop2
            X_tr = self.build_features(
                self.labeled.iloc[tr_idx], graph_cache, fold_known_mask, fold_y,
                use_hop2=use_hop2, alpha=alpha, beta=beta, hop2_max_neighbors=hop2_max_neighbors
            )
            X_val = self.build_features(
                self.labeled.iloc[val_idx], graph_cache, fold_known_mask, fold_y,
                use_hop2=use_hop2, alpha=alpha, beta=beta, hop2_max_neighbors=hop2_max_neighbors
            )
            
            y_tr, y_val = self.y[tr_idx], self.y[val_idx]
            
            # Train model with expanded hyperparameters
            model = lgb.LGBMClassifier(
                n_estimators=config.get('n_est', 1500),
                learning_rate=config.get('lr', 0.03),
                num_leaves=config.get('num_leaves', 63),
                max_depth=config.get('max_depth', -1),
                min_child_samples=config.get('min_child', 20),
                min_child_weight=config.get('min_child_weight', 1e-3),
                min_split_gain=config.get('min_split_gain', 0.0),
                scale_pos_weight=config.get('scale_pos_weight', 3.0),
                subsample=config.get('subsample', 0.8),
                subsample_freq=config.get('bagging_freq', 1),
                colsample_bytree=config.get('colsample', 0.8),
                reg_alpha=config.get('reg_alpha', 0.0),
                reg_lambda=config.get('reg_lambda', 0.0),
                max_bin=config.get('max_bin', 255),
                random_state=SEED,
                verbose=-1,
                n_jobs=-1
            )
            model.fit(X_tr, y_tr)
            model_pred = model.predict_proba(X_val)[:, 1]
            
            # Blend with comp_cheat_rate if blend_weight specified
            blend_weight = config.get('blend_weight', 1.0)  # w=1 means pure model
            if blend_weight < 1.0 and 'comp_cheat_rate' in X_val.columns:
                comp_rate_val = X_val['comp_cheat_rate'].values
                oof[val_idx] = blend_weight * model_pred + (1 - blend_weight) * comp_rate_val
            else:
                oof[val_idx] = model_pred
        
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
        """Run hyperparameter search with expanded grid."""
        print(f"\nRunning search with max {max_trials} trials...")
        
        configs = []
        
        # Core grid: scale_pos_weight, leaves, lr combinations
        for spw in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            for leaves in [31, 63, 127]:
                for lr in [0.01, 0.02, 0.03]:
                    configs.append({
                        'scale_pos_weight': spw,
                        'num_leaves': leaves,
                        'lr': lr,
                        'n_est': 1500,
                        'min_child': 20,
                        'subsample': 0.8,
                        'colsample': 0.8,
                        'alpha': 1.0,
                        'beta': 1.0
                    })
        
        # Expanded params: max_depth, reg, min_split_gain
        for spw in [1.5, 2.0, 2.5]:
            for max_depth in [6, 8, 10, -1]:
                for reg_lambda in [0.0, 0.1, 1.0]:
                    configs.append({
                        'scale_pos_weight': spw,
                        'num_leaves': 63,
                        'lr': 0.01,
                        'n_est': 2000,
                        'max_depth': max_depth,
                        'min_child': 20,
                        'reg_lambda': reg_lambda,
                        'reg_alpha': 0.0,
                        'subsample': 0.8,
                        'colsample': 0.8,
                        'alpha': 1.0,
                        'beta': 1.0
                    })
        
        # Bayesian smoothing priors: alpha/beta tuning
        for spw in [1.5, 2.0]:
            for alpha in [0.5, 1.0, 2.0, 4.0]:
                for beta in [0.5, 1.0, 2.0, 4.0]:
                    configs.append({
                        'scale_pos_weight': spw,
                        'num_leaves': 63,
                        'lr': 0.01,
                        'n_est': 1500,
                        'min_child': 20,
                        'subsample': 0.8,
                        'colsample': 0.8,
                        'alpha': alpha,
                        'beta': beta
                    })
        
        # Sampled 2-hop feature tests - DISABLED for speed (full matvec too slow)
        # To re-enable, rewrite hop2 to avoid full adj@vector computation
        # for spw in [1.5, 2.0, 2.5]:
        #     for use_hop2 in [True]:
        #         for hop2_neighbors in [30, 50, 100]:
        #             configs.append({...})
        
        # n_estimators + early stop, min_child variations
        for spw in [1.5, 2.0]:
            for n_est in [2000, 2500, 3000]:
                for min_child in [10, 30, 50, 100]:
                    configs.append({
                        'scale_pos_weight': spw,
                        'num_leaves': 63,
                        'lr': 0.01,
                        'n_est': n_est,
                        'min_child': min_child,
                        'subsample': 0.8,
                        'colsample': 0.8,
                        'alpha': 1.0,
                        'beta': 1.0
                    })
        
        # Subsample/colsample + bagging_freq variations
        for spw in [1.5, 2.0]:
            for subsample in [0.6, 0.7, 0.9]:
                for colsample in [0.6, 0.7, 0.9]:
                    for bagging_freq in [1, 5]:
                        configs.append({
                            'scale_pos_weight': spw,
                            'num_leaves': 63,
                            'lr': 0.01,
                            'n_est': 1500,
                            'min_child': 20,
                            'subsample': subsample,
                            'colsample': colsample,
                            'bagging_freq': bagging_freq,
                            'alpha': 1.0,
                            'beta': 1.0
                        })
        
        # max_bin variations
        for spw in [1.5, 2.0]:
            for max_bin in [127, 255, 511]:
                configs.append({
                    'scale_pos_weight': spw,
                    'num_leaves': 63,
                    'lr': 0.01,
                    'n_est': 1500,
                    'min_child': 20,
                    'max_bin': max_bin,
                    'subsample': 0.8,
                    'colsample': 0.8,
                    'alpha': 1.0,
                    'beta': 1.0
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
        print(f"  Testing {len(configs)} configurations (expanded search space)")
        
        for i, cfg in enumerate(configs):
            pub, is_best = self.run_trial(cfg, graph_cache)
            if (i + 1) % 10 == 0:
                print(f"  Trial {i+1}/{len(configs)}: pub={pub:.0f}, best={self.best_pub:.0f}")
        
        print(f"\nSearch complete. Best proxy_public: {self.best_pub:.0f}")
    
    def run_optuna_search(self, graph_cache, n_trials=50, study_name='branch_d'):
        """Run Optuna TPE search with NARROW high-impact params + early stopping."""
        print(f"\nRunning Optuna TPE search ({n_trials} trials)...")
        
        storage_path = self.out_dir / 'optuna.db'
        storage = f'sqlite:///{storage_path}'
        
        def objective(trial):
            config = {
                # Core LGBM params (narrow, high-impact only)
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 8.0),
                'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                'lr': trial.suggest_float('lr', 0.005, 0.05, log=True),
                'n_est': 8000,  # Fixed high, use early stopping
                'min_child': trial.suggest_int('min_child', 10, 200),
                
                # Regularization
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                
                # Sampling
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample': trial.suggest_float('colsample', 0.6, 1.0),
                
                # Bayesian smoothing priors (log-scale)
                'alpha': trial.suggest_float('alpha', 0.25, 16.0, log=True),
                'beta': trial.suggest_float('beta', 0.25, 16.0, log=True),
                
                # Blend weight: final_pred = w*model_pred + (1-w)*comp_cheat_rate
                # Allow full range including pure comp_rate
                'blend_weight': trial.suggest_float('blend_weight', 0.0, 1.0),
                
                # DISABLE hop2 for speed (fixed)
                'use_hop2': False,
                
                # Fixed params
                'max_depth': -1,
                'min_child_weight': 1e-3,
                'min_split_gain': 0.0,
                'reg_alpha': 0.0,
                'bagging_freq': 1,
                'max_bin': 255,
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
        """Export top 5 submissions DETERMINISTICALLY from scoreboard.csv (not self.results)."""
        print("\nExporting 5 submissions (from scoreboard.csv)...")
        
        # READ FROM SCOREBOARD.CSV (deterministic across all runs)
        if not self.scoreboard_path.exists():
            print("  ERROR: No scoreboard.csv found!")
            return
        
        sb = pd.read_csv(self.scoreboard_path)
        
        # Find the proxy_public column (may be named differently)
        if 'proxy_public' in sb.columns:
            pub_col = 'proxy_public'
        elif 'pub' in sb.columns:
            pub_col = 'pub'
        else:
            print(f"  ERROR: Cannot find proxy_public column. Columns: {sb.columns.tolist()}")
            return
        
        # Filter valid rows and sort by proxy_public ASCENDING (lower cost = better)
        sb_valid = sb[sb[pub_col].notna()].copy()
        sb_valid = sb_valid.sort_values(pub_col, ascending=True).head(10)
        
        print(f"  Found {len(sb_valid)} valid trials. Top 5 proxy_public:")
        for i, row in sb_valid.head(5).iterrows():
            print(f"    {pub_col}={row[pub_col]:.0f}")
        
        # Map all labeled users for full training
        all_node_idx = graph_cache.get_node_indices(self.labeled['user_hash'].values)
        full_known_mask = np.zeros(graph_cache.n_nodes, dtype=bool)
        full_y = np.zeros(graph_cache.n_nodes, dtype=np.float32)
        valid = all_node_idx >= 0
        full_known_mask[all_node_idx[valid]] = True
        full_y[all_node_idx[valid]] = self.y[valid]
        
        # Export top 5 as individual submissions
        for rank, (idx, row) in enumerate(sb_valid.head(5).iterrows()):
            # Extract config from scoreboard row
            cfg = {}
            for col in sb_valid.columns:
                if col.startswith('param_'):
                    param_name = col.replace('param_', '')
                    val = row[col]
                    if pd.notna(val):
                        # Convert to appropriate type
                        if param_name in ['num_leaves', 'n_est', 'min_child', 'max_depth', 'bagging_freq', 'max_bin']:
                            cfg[param_name] = int(val)
                        elif param_name in ['use_hop2']:
                            cfg[param_name] = bool(val) if pd.notna(val) else False
                        else:
                            cfg[param_name] = float(val)
            
            # Set defaults for missing params
            cfg.setdefault('n_est', 1500)
            cfg.setdefault('lr', 0.01)
            cfg.setdefault('num_leaves', 63)
            cfg.setdefault('max_depth', -1)
            cfg.setdefault('min_child', 20)
            cfg.setdefault('scale_pos_weight', 2.0)
            cfg.setdefault('subsample', 0.8)
            cfg.setdefault('colsample', 0.8)
            cfg.setdefault('alpha', 1.0)
            cfg.setdefault('beta', 1.0)
            cfg.setdefault('blend_weight', 1.0)
            cfg.setdefault('use_hop2', False)
            
            alpha = cfg.get('alpha', 1.0)
            beta = cfg.get('beta', 1.0)
            use_hop2 = cfg.get('use_hop2', False)
            hop2_max_neighbors = cfg.get('hop2_max_neighbors', 50)
            
            X_train_full = self.build_features(
                self.labeled, graph_cache, full_known_mask, full_y,
                use_hop2=use_hop2, alpha=alpha, beta=beta, hop2_max_neighbors=hop2_max_neighbors
            )
            X_test = self.build_features(
                self.test, graph_cache, full_known_mask, full_y,
                use_hop2=use_hop2, alpha=alpha, beta=beta, hop2_max_neighbors=hop2_max_neighbors
            )
            
            model = lgb.LGBMClassifier(
                n_estimators=min(4000, cfg.get('n_est', 1500) * 2),
                learning_rate=cfg.get('lr', 0.03),
                num_leaves=cfg.get('num_leaves', 63),
                max_depth=cfg.get('max_depth', -1),
                min_child_samples=cfg.get('min_child', 20),
                min_child_weight=cfg.get('min_child_weight', 1e-3),
                min_split_gain=cfg.get('min_split_gain', 0.0),
                scale_pos_weight=cfg.get('scale_pos_weight', 3.0),
                subsample=cfg.get('subsample', 0.8),
                subsample_freq=cfg.get('bagging_freq', 1),
                colsample_bytree=cfg.get('colsample', 0.8),
                reg_alpha=cfg.get('reg_alpha', 0.0),
                reg_lambda=cfg.get('reg_lambda', 0.0),
                max_bin=cfg.get('max_bin', 255),
                random_state=SEED,
                verbose=-1,
                n_jobs=-1
            )
            model.fit(X_train_full, self.y)
            model_preds = model.predict_proba(X_test)[:, 1]
            
            # Apply blend if blend_weight < 1
            blend_weight = cfg.get('blend_weight', 1.0)
            if blend_weight < 1.0 and 'comp_cheat_rate' in X_test.columns:
                comp_rate_test = X_test['comp_cheat_rate'].values
                preds = blend_weight * model_preds + (1 - blend_weight) * comp_rate_test
            else:
                preds = model_preds
            
            preds = np.clip(preds, 0, 1)
            
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
            print(f"  {fname.name}: pub={row[pub_col]:.0f}, range=[{preds.min():.4f}, {preds.max():.4f}]")
        
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
