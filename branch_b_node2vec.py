"""
Branch B: Node2Vec/DeepWalk Embeddings + GBDT
=============================================
Exploit social_graph with graph embeddings combined with tabular features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
from gensim.models import Word2Vec
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from data_loader import (
    SEED, load_train_data, load_test_data, load_graph,
    get_feature_columns, create_features, split_labeled_unlabeled,
    create_proxy_split, set_seed
)
from cv_harness import (
    cv_train_evaluate, train_final_model, evaluate_on_subset,
    save_cv_artifacts
)
import metric


def build_graph(edge_df: pd.DataFrame) -> nx.Graph:
    """Build NetworkX graph from edge list."""
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row['user_a'], row['user_b'])
    return G


def compute_graph_features_fast(
    G: nx.Graph,
    user_hashes: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute fast graph features as alternative to Node2Vec for large graphs.
    Includes degree-based features and simple structural features.
    """
    features = []
    
    # Pre-compute some statistics
    all_degrees = dict(G.degree())
    
    for i, uh in enumerate(user_hashes):
        if i % 50000 == 0 and verbose:
            print(f"    Processing {i}/{len(user_hashes)}...")
        
        if uh in G:
            degree = all_degrees[uh]
            neighbors = list(G.neighbors(uh))
            
            if neighbors:
                neighbor_degrees = [all_degrees.get(n, 0) for n in neighbors]
                feat = [
                    degree,
                    np.log1p(degree),
                    np.mean(neighbor_degrees),
                    np.max(neighbor_degrees),
                    np.min(neighbor_degrees),
                    np.std(neighbor_degrees) if len(neighbor_degrees) > 1 else 0,
                    len(neighbors),
                    sum(1 for n in neighbors if all_degrees.get(n, 0) > degree),  # neighbors with higher degree
                    sum(1 for n in neighbors if all_degrees.get(n, 0) < degree),  # neighbors with lower degree
                ]
            else:
                feat = [degree, np.log1p(degree), 0, 0, 0, 0, 0, 0, 0]
        else:
            feat = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        features.append(feat)
    
    return np.array(features)


def create_branch_b_features(
    df: pd.DataFrame,
    G: nx.Graph,
    graph_features: np.ndarray = None
) -> np.ndarray:
    """Create features combining tabular + graph features + degree."""
    # Tabular features
    tabular = create_features(df, add_missing_flags=True, add_row_stats=True).values
    
    # Combine
    features = np.hstack([tabular, graph_features])
    return features


def run_branch_b(
    output_dir: str = 'artifacts/branch_B_node2vec',
    verbose: bool = True,
    quick_mode: bool = False
) -> Dict:
    """
    Run Branch B: Graph Features + GBDT pipeline.
    
    Uses fast graph features instead of Node2Vec for scalability.
    
    Args:
        output_dir: Directory for artifacts
        verbose: Print progress
        quick_mode: Use fewer iterations for testing
    
    Returns:
        Results dictionary
    """
    set_seed(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("="*60)
        print("BRANCH B: GRAPH FEATURES + GBDT")
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
    
    if verbose:
        print(f"\nLabeled samples: {len(labeled):,}")
        print(f"Unlabeled samples: {len(unlabeled):,}")
    
    # Compute graph features for all datasets
    if verbose:
        print("\nComputing graph features for labeled data...")
    graph_feat_labeled = compute_graph_features_fast(G, labeled['user_hash'].values, verbose=verbose)
    
    if verbose:
        print("\nComputing graph features for unlabeled data...")
    graph_feat_unlabeled = compute_graph_features_fast(G, unlabeled['user_hash'].values, verbose=verbose)
    
    if verbose:
        print("\nComputing graph features for test data...")
    graph_feat_test = compute_graph_features_fast(G, test['user_hash'].values, verbose=verbose)
    
    # Create features
    if verbose:
        print("\nCreating combined features...")
    X_labeled = create_branch_b_features(labeled, G, graph_feat_labeled)
    X_unlabeled = create_branch_b_features(unlabeled, G, graph_feat_unlabeled)
    X_test = create_branch_b_features(test, G, graph_feat_test)
    y_labeled = labeled['is_cheating'].values
    
    # Handle NaN
    X_labeled = np.nan_to_num(X_labeled, nan=-999)
    X_unlabeled = np.nan_to_num(X_unlabeled, nan=-999)
    X_test = np.nan_to_num(X_test, nan=-999)
    
    if verbose:
        print(f"Feature shape: {X_labeled.shape}")
    
    # Create proxy split
    public_proxy, private_proxy = create_proxy_split(labeled, seed=SEED)
    public_idx = labeled['user_hash'].isin(public_proxy['user_hash']).values
    private_idx = labeled['user_hash'].isin(private_proxy['user_hash']).values
    
    # Train with CV
    n_estimators = 100 if quick_mode else 500
    
    if verbose:
        print("\n--- LightGBM with Node2Vec ---")
    
    lgbm_oof, lgbm_summary = cv_train_evaluate(
        X_labeled, y_labeled, X_unlabeled,
        lambda: LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbosity=-1
        ),
        n_folds=5,
        pseudo_weight=0.01,
        verbose=verbose
    )
    
    # Evaluate on proxy
    public_result = evaluate_on_subset(y_labeled, lgbm_oof, public_idx, "public_proxy")
    private_result = evaluate_on_subset(y_labeled, lgbm_oof, private_idx, "private_proxy")
    
    if verbose:
        print(f"\nProxy evaluation:")
        print(f"  Public proxy cost:  {public_result['best_cost']:,}")
        print(f"  Private proxy cost: {private_result['best_cost']:,}")
    
    # Train final model
    if verbose:
        print("\n--- Training final model ---")
    
    final_model = train_final_model(
        X_labeled, y_labeled, X_unlabeled,
        lambda: LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbosity=-1
        ),
        pseudo_weight=0.01
    )
    
    # Get test predictions
    test_preds = final_model.predict_proba(X_test)[:, 1]
    
    # Save artifacts
    save_cv_artifacts(
        lgbm_oof,
        labeled['user_hash'].values,
        lgbm_summary,
        output_dir
    )
    
    with open(output_dir / 'proxy_public.json', 'w') as f:
        json.dump({
            'cost': public_result['best_cost'],
            'score': public_result['best_score'],
            't_low': public_result['t_low'],
            't_high': public_result['t_high']
        }, f, indent=2)
    
    with open(output_dir / 'proxy_private.json', 'w') as f:
        json.dump({
            'cost': private_result['best_cost'],
            'score': private_result['best_score'],
            't_low': private_result['t_low'],
            't_high': private_result['t_high']
        }, f, indent=2)
    
    # Create submission
    submission = pd.DataFrame({
        'user_hash': test['user_hash'],
        'prediction': test_preds
    })
    submission.to_csv('submission_B.csv', index=False)
    
    if verbose:
        print(f"\nSubmission saved to: submission_B.csv")
        print(f"Prediction range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    
    oof_eval = metric.score(y_labeled, lgbm_oof)
    
    results = {
        'branch': 'B',
        'method': 'Graph Features + GBDT',
        'oof_cost': lgbm_summary['oof_cost'],
        'oof_score': lgbm_summary['oof_score'],
        'public_proxy_cost': public_result['best_cost'],
        'private_proxy_cost': private_result['best_cost'],
        't_low': oof_eval['t_low'],
        't_high': oof_eval['t_high'],
        'submission_file': 'submission_B.csv'
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print("\n" + "="*60)
        print("BRANCH B COMPLETE")
        print(f"  OOF Cost: {lgbm_summary['oof_cost']:,}")
        print(f"  Public Proxy Cost: {public_result['best_cost']:,}")
        print(f"  Private Proxy Cost: {private_result['best_cost']:,}")
        print("="*60)
    
    return results


if __name__ == '__main__':
    results = run_branch_b(verbose=True, quick_mode=False)
    print(json.dumps(results, indent=2))
