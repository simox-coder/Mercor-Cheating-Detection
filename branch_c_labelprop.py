"""
Branch C: Fold-Safe Label Propagation + GBDT
============================================
Semi-supervised graph signal using ONLY fold-train labels.
CRITICAL: No leakage from fold-val labels into propagation features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from lightgbm import LGBMClassifier

from data_loader import (
    SEED, load_train_data, load_test_data, load_graph,
    create_features, split_labeled_unlabeled, create_proxy_split, set_seed
)
from cv_harness import create_cv_folds, evaluate_on_subset, save_cv_artifacts
import metric


def build_graph(edge_df: pd.DataFrame) -> nx.Graph:
    """Build NetworkX graph from edge list."""
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row['user_a'], row['user_b'])
    return G


def compute_personalized_pagerank(
    G: nx.Graph,
    seed_scores: Dict[str, float],
    alpha: float = 0.85,
    max_iter: int = 100
) -> Dict[str, float]:
    """
    Compute Personalized PageRank from seed nodes.
    
    Args:
        G: NetworkX graph
        seed_scores: Dictionary mapping node -> seed score
        alpha: Damping factor
        max_iter: Maximum iterations
    
    Returns:
        Dictionary mapping node -> PPR score
    """
    # Normalize seed scores to sum to 1
    total = sum(seed_scores.values())
    if total == 0:
        return {node: 0.0 for node in G.nodes()}
    
    personalization = {node: seed_scores.get(node, 0) / total for node in G.nodes()}
    
    try:
        ppr = nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=max_iter)
    except:
        # Fallback to uniform
        ppr = {node: 1.0 / G.number_of_nodes() for node in G.nodes()}
    
    return ppr


def compute_label_propagation_scores(
    G: nx.Graph,
    labeled_nodes: Dict[str, float],
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Simple label propagation from labeled nodes.
    
    Args:
        G: NetworkX graph  
        labeled_nodes: Dictionary mapping node -> label (0 or 1)
        num_iterations: Number of propagation iterations
    
    Returns:
        Dictionary mapping all nodes -> propagated score
    """
    # Initialize scores
    scores = {}
    for node in G.nodes():
        if node in labeled_nodes:
            scores[node] = labeled_nodes[node]
        else:
            scores[node] = 0.5  # Unknown
    
    labeled_set = set(labeled_nodes.keys())
    
    # Propagate
    for _ in range(num_iterations):
        new_scores = {}
        for node in G.nodes():
            if node in labeled_set:
                new_scores[node] = labeled_nodes[node]  # Keep labeled fixed
            else:
                neighbors = list(G.neighbors(node))
                if neighbors:
                    new_scores[node] = np.mean([scores[n] for n in neighbors])
                else:
                    new_scores[node] = 0.5
        scores = new_scores
    
    return scores


def get_neighbor_stats_fast(
    G: nx.Graph,
    user_hashes: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """Get fast neighbor statistics for multiple nodes."""
    all_degrees = dict(G.degree())
    features = []
    
    for i, node in enumerate(user_hashes):
        if i % 50000 == 0 and verbose:
            print(f"    Processing {i}/{len(user_hashes)}...")
        
        if node not in G:
            features.append([0, 0, 0, 0])
            continue
        
        degree = all_degrees.get(node, 0)
        neighbors = list(G.neighbors(node))
        
        if neighbors:
            neighbor_degrees = [all_degrees.get(n, 0) for n in neighbors]
            neighbor_mean = np.mean(neighbor_degrees)
            neighbor_max = np.max(neighbor_degrees)
        else:
            neighbor_mean = 0
            neighbor_max = 0
        
        features.append([
            degree,
            np.log1p(degree),
            neighbor_mean,
            neighbor_max
        ])
    
    return np.array(features)


def run_branch_c(
    output_dir: str = 'artifacts/branch_C_labelprop',
    verbose: bool = True,
    quick_mode: bool = False
) -> Dict:
    """
    Run Branch C: Label Propagation + GBDT pipeline.
    
    CRITICAL: Propagation features computed per-fold using only fold-train labels.
    """
    set_seed(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("="*60)
        print("BRANCH C: FOLD-SAFE LABEL PROPAGATION + GBDT")
        print("="*60)
    
    # Load data
    train = load_train_data()
    test = load_test_data()
    graph_df = load_graph()
    
    # Build graph
    if verbose:
        print("Building graph...")
    G = build_graph(graph_df)
    if verbose:
        print(f"  Nodes: {G.number_of_nodes():,}")
        print(f"  Edges: {G.number_of_edges():,}")
    
    # Split labeled/unlabeled
    labeled, unlabeled = split_labeled_unlabeled(train)
    y_labeled = labeled['is_cheating'].values
    labeled_hashes = labeled['user_hash'].values
    
    if verbose:
        print(f"\nLabeled samples: {len(labeled):,}")
        print(f"Unlabeled samples: {len(unlabeled):,}")
    
    # Create base tabular features
    if verbose:
        print("\nCreating base features...")
    
    X_tabular_labeled = create_features(labeled, add_missing_flags=True, add_row_stats=True).values
    X_tabular_unlabeled = create_features(unlabeled, add_missing_flags=True, add_row_stats=True).values
    X_tabular_test = create_features(test, add_missing_flags=True, add_row_stats=True).values
    
    # Pre-compute neighbor stats (these don't depend on labels)
    if verbose:
        print("Computing neighbor statistics...")
    
    neighbor_feat_labeled = get_neighbor_stats_fast(G, labeled_hashes, verbose=verbose)
    neighbor_feat_unlabeled = get_neighbor_stats_fast(G, unlabeled['user_hash'].values, verbose=verbose)
    neighbor_feat_test = get_neighbor_stats_fast(G, test['user_hash'].values, verbose=verbose)
    
    # Create proxy split (for final evaluation)
    public_proxy, private_proxy = create_proxy_split(labeled, seed=SEED)
    public_idx = labeled['user_hash'].isin(public_proxy['user_hash']).values
    private_idx = labeled['user_hash'].isin(private_proxy['user_hash']).values
    
    # CV with fold-safe propagation
    if verbose:
        print("\n--- Fold-safe CV with label propagation ---")
    
    folds = create_cv_folds(y_labeled, n_folds=5, seed=SEED)
    oof_preds = np.zeros(len(y_labeled))
    fold_costs = []
    
    n_estimators = 100 if quick_mode else 500
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if verbose:
            print(f"\n  Fold {fold_idx + 1}/5:")
        
        # Get fold-train labels ONLY
        fold_train_labels = {}
        for idx in train_idx:
            node = labeled_hashes[idx]
            fold_train_labels[node] = y_labeled[idx]
        
        # Compute propagation features using ONLY fold-train labels
        if verbose:
            print("    Computing PPR from positive seeds...")
        
        positive_seeds = {n: 1.0 for n, y in fold_train_labels.items() if y == 1}
        negative_seeds = {n: 1.0 for n, y in fold_train_labels.items() if y == 0}
        
        ppr_positive = compute_personalized_pagerank(G, positive_seeds, alpha=0.85)
        ppr_negative = compute_personalized_pagerank(G, negative_seeds, alpha=0.85)
        
        # Also run simple label propagation
        if verbose:
            print("    Computing label propagation...")
        lp_scores = compute_label_propagation_scores(G, fold_train_labels, num_iterations=10)
        
        # Extract propagation features for all samples
        def get_prop_features(user_hashes):
            feats = []
            for uh in user_hashes:
                ppr_p = ppr_positive.get(uh, 0)
                ppr_n = ppr_negative.get(uh, 0)
                lp = lp_scores.get(uh, 0.5)
                # Combine into feature
                ppr_diff = ppr_p - ppr_n
                ppr_ratio = ppr_p / (ppr_p + ppr_n + 1e-10)
                feats.append([ppr_p, ppr_n, ppr_diff, ppr_ratio, lp])
            return np.array(feats)
        
        prop_feat_labeled = get_prop_features(labeled_hashes)
        prop_feat_unlabeled = get_prop_features(unlabeled['user_hash'].values)
        
        # Combine all features
        X_full_labeled = np.hstack([
            X_tabular_labeled, 
            neighbor_feat_labeled,
            prop_feat_labeled
        ])
        X_full_unlabeled = np.hstack([
            X_tabular_unlabeled,
            neighbor_feat_unlabeled,
            prop_feat_unlabeled
        ])
        
        # Handle NaN
        X_full_labeled = np.nan_to_num(X_full_labeled, nan=-999)
        X_full_unlabeled = np.nan_to_num(X_full_unlabeled, nan=-999)
        
        # Train/val split
        X_train = X_full_labeled[train_idx]
        y_train = y_labeled[train_idx]
        X_val = X_full_labeled[val_idx]
        y_val = y_labeled[val_idx]
        
        # Add pseudo-negatives
        X_train_full = np.vstack([X_train, X_full_unlabeled])
        y_train_full = np.concatenate([y_train, np.zeros(len(X_full_unlabeled))])
        weights = np.concatenate([
            np.ones(len(X_train)),
            np.full(len(X_full_unlabeled), 0.01)
        ])
        
        # Train model
        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbosity=-1
        )
        model.fit(X_train_full, y_train_full, sample_weight=weights)
        
        # Predict
        fold_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = fold_preds
        
        # Evaluate
        fold_result = metric.score(y_val, fold_preds)
        fold_costs.append(fold_result['best_cost'])
        if verbose:
            print(f"    Fold cost: {fold_result['best_cost']:,}")
    
    # Overall OOF evaluation
    oof_result = metric.score(y_labeled, oof_preds)
    
    if verbose:
        print(f"\n  OOF Cost: {oof_result['best_cost']:,}")
    
    # Evaluate on proxy
    public_result = evaluate_on_subset(y_labeled, oof_preds, public_idx, "public_proxy")
    private_result = evaluate_on_subset(y_labeled, oof_preds, private_idx, "private_proxy")
    
    if verbose:
        print(f"\nProxy evaluation:")
        print(f"  Public proxy cost:  {public_result['best_cost']:,}")
        print(f"  Private proxy cost: {private_result['best_cost']:,}")
    
    # Train final model using all labeled data for propagation
    if verbose:
        print("\n--- Training final model ---")
    
    # Final propagation using ALL labeled data
    all_labeled_nodes = {labeled_hashes[i]: y_labeled[i] for i in range(len(y_labeled))}
    
    positive_seeds_final = {n: 1.0 for n, y in all_labeled_nodes.items() if y == 1}
    negative_seeds_final = {n: 1.0 for n, y in all_labeled_nodes.items() if y == 0}
    
    ppr_positive_final = compute_personalized_pagerank(G, positive_seeds_final)
    ppr_negative_final = compute_personalized_pagerank(G, negative_seeds_final)
    lp_scores_final = compute_label_propagation_scores(G, all_labeled_nodes)
    
    def get_final_prop_features(user_hashes):
        feats = []
        for uh in user_hashes:
            ppr_p = ppr_positive_final.get(uh, 0)
            ppr_n = ppr_negative_final.get(uh, 0)
            lp = lp_scores_final.get(uh, 0.5)
            ppr_diff = ppr_p - ppr_n
            ppr_ratio = ppr_p / (ppr_p + ppr_n + 1e-10)
            feats.append([ppr_p, ppr_n, ppr_diff, ppr_ratio, lp])
        return np.array(feats)
    
    prop_feat_labeled_final = get_final_prop_features(labeled_hashes)
    prop_feat_unlabeled_final = get_final_prop_features(unlabeled['user_hash'].values)
    prop_feat_test_final = get_final_prop_features(test['user_hash'].values)
    
    X_full_labeled_final = np.hstack([
        X_tabular_labeled,
        neighbor_feat_labeled,
        prop_feat_labeled_final
    ])
    X_full_unlabeled_final = np.hstack([
        X_tabular_unlabeled,
        neighbor_feat_unlabeled,
        prop_feat_unlabeled_final
    ])
    X_full_test_final = np.hstack([
        X_tabular_test,
        neighbor_feat_test,
        prop_feat_test_final
    ])
    
    X_full_labeled_final = np.nan_to_num(X_full_labeled_final, nan=-999)
    X_full_unlabeled_final = np.nan_to_num(X_full_unlabeled_final, nan=-999)
    X_full_test_final = np.nan_to_num(X_full_test_final, nan=-999)
    
    # Train final
    X_train_final = np.vstack([X_full_labeled_final, X_full_unlabeled_final])
    y_train_final = np.concatenate([y_labeled, np.zeros(len(X_full_unlabeled_final))])
    weights_final = np.concatenate([
        np.ones(len(y_labeled)),
        np.full(len(X_full_unlabeled_final), 0.01)
    ])
    
    final_model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbosity=-1
    )
    final_model.fit(X_train_final, y_train_final, sample_weight=weights_final)
    
    # Test predictions
    test_preds = final_model.predict_proba(X_full_test_final)[:, 1]
    
    # Save artifacts
    summary = {
        'fold_costs': fold_costs,
        'mean_fold_cost': float(np.mean(fold_costs)),
        'std_fold_cost': float(np.std(fold_costs)),
        'oof_cost': oof_result['best_cost'],
        'oof_score': oof_result['best_score'],
        't_low': oof_result['t_low'],
        't_high': oof_result['t_high']
    }
    
    save_cv_artifacts(oof_preds, labeled_hashes, summary, output_dir)
    
    with open(output_dir / 'proxy_public.json', 'w') as f:
        json.dump({
            'cost': public_result['best_cost'],
            'score': public_result['best_score']
        }, f, indent=2)
    
    with open(output_dir / 'proxy_private.json', 'w') as f:
        json.dump({
            'cost': private_result['best_cost'],
            'score': private_result['best_score']
        }, f, indent=2)
    
    # Create submission
    submission = pd.DataFrame({
        'user_hash': test['user_hash'],
        'prediction': test_preds
    })
    submission.to_csv('submission_C.csv', index=False)
    
    if verbose:
        print(f"\nSubmission saved to: submission_C.csv")
        print(f"Prediction range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    
    results = {
        'branch': 'C',
        'method': 'Fold-Safe Label Propagation + GBDT',
        'oof_cost': oof_result['best_cost'],
        'oof_score': oof_result['best_score'],
        'public_proxy_cost': public_result['best_cost'],
        'private_proxy_cost': private_result['best_cost'],
        't_low': oof_result['t_low'],
        't_high': oof_result['t_high'],
        'submission_file': 'submission_C.csv'
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print("\n" + "="*60)
        print("BRANCH C COMPLETE")
        print(f"  OOF Cost: {oof_result['best_cost']:,}")
        print(f"  Public Proxy Cost: {public_result['best_cost']:,}")
        print(f"  Private Proxy Cost: {private_result['best_cost']:,}")
        print("="*60)
    
    return results


if __name__ == '__main__':
    results = run_branch_c(verbose=True, quick_mode=False)
    print(json.dumps(results, indent=2))
