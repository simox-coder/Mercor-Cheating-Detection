"""
Mercor Cheating Detection - Experiment Framework

This module provides:
- Experiment logging to artifacts/exp_log.csv
- Public/private proxy split evaluation
- Iterative experiment tracking with verified metrics
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import metric

# Configuration
SEED = 42
N_FOLDS = 5
N_THRESHOLDS = 300
ARTIFACTS_DIR = "artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def create_proxy_split(labeled_df: pd.DataFrame, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create public/private proxy split (50/50, stratified).
    
    This simulates the public/private leaderboard split for local validation.
    """
    public_proxy, private_proxy = train_test_split(
        labeled_df,
        test_size=0.5,
        stratify=labeled_df['is_cheating'],
        random_state=seed
    )
    
    print(f"Proxy split:")
    print(f"  Public:  {len(public_proxy):,} ({100*public_proxy['is_cheating'].mean():.1f}% cheaters)")
    print(f"  Private: {len(private_proxy):,} ({100*private_proxy['is_cheating'].mean():.1f}% cheaters)")
    
    return public_proxy.reset_index(drop=True), private_proxy.reset_index(drop=True)


def evaluate_on_proxy(
    oof_df: pd.DataFrame, 
    public_proxy: pd.DataFrame, 
    private_proxy: pd.DataFrame,
    pred_col: str = 'prediction'
) -> Dict:
    """
    Evaluate OOF predictions on public and private proxy splits.
    
    Returns metrics for both splits.
    """
    # Match predictions to proxy splits by user_hash
    public_merged = public_proxy.merge(
        oof_df[['user_hash', pred_col]], 
        on='user_hash', 
        how='left'
    )
    private_merged = private_proxy.merge(
        oof_df[['user_hash', pred_col]], 
        on='user_hash', 
        how='left'
    )
    
    # Evaluate on public proxy
    pub_result = metric.evaluate(
        public_merged['is_cheating'].values,
        public_merged[pred_col].values,
        n_thresholds=N_THRESHOLDS
    )
    
    # Evaluate on private proxy
    priv_result = metric.evaluate(
        private_merged['is_cheating'].values,
        private_merged[pred_col].values,
        n_thresholds=N_THRESHOLDS
    )
    
    return {
        'public_cost': pub_result.best_cost,
        'public_score': pub_result.best_score,
        'public_t_low': pub_result.t_low,
        'public_t_high': pub_result.t_high,
        'private_cost': priv_result.best_cost,
        'private_score': priv_result.best_score,
        'private_t_low': priv_result.t_low,
        'private_t_high': priv_result.t_high,
    }


def log_experiment(
    exp_id: str,
    exp_name: str,
    cv_cost: int,
    public_cost: int,
    private_cost: int,
    cv_t_low: float,
    cv_t_high: float,
    notes: str = "",
    extra_metrics: Dict = None
):
    """
    Log experiment results to artifacts/exp_log.csv
    """
    log_path = os.path.join(ARTIFACTS_DIR, "exp_log.csv")
    
    # Create header if file doesn't exist
    file_exists = os.path.exists(log_path)
    
    row = {
        'exp_id': exp_id,
        'timestamp': datetime.now().isoformat(),
        'exp_name': exp_name,
        'cv_cost': cv_cost,
        'cv_score': -cv_cost,
        'public_cost': public_cost,
        'public_score': -public_cost,
        'private_cost': private_cost,
        'private_score': -private_cost,
        'cv_t_low': cv_t_low,
        'cv_t_high': cv_t_high,
        'stability_diff_pct': 100 * (public_cost - private_cost) / private_cost if private_cost > 0 else 0,
        'notes': notes,
    }
    
    if extra_metrics:
        row.update(extra_metrics)
    
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"\nExperiment {exp_id} logged:")
    print(f"  CV Cost: {cv_cost:,} | Public: {public_cost:,} | Private: {private_cost:,}")
    print(f"  Stability diff: {row['stability_diff_pct']:.1f}%")


def save_experiment_artifacts(
    exp_id: str,
    oof_df: pd.DataFrame,
    cv_summary: Dict,
    model_config: Dict = None
):
    """
    Save experiment artifacts to artifacts/exp_<id>/
    """
    exp_dir = os.path.join(ARTIFACTS_DIR, f"exp_{exp_id}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save OOF predictions
    oof_df.to_csv(os.path.join(exp_dir, "oof.csv"), index=False)
    
    # Save CV summary
    with open(os.path.join(exp_dir, "cv_summary.json"), 'w') as f:
        json.dump(cv_summary, f, indent=2, default=str)
    
    # Save model config
    if model_config:
        with open(os.path.join(exp_dir, "config.json"), 'w') as f:
            json.dump(model_config, f, indent=2)
    
    print(f"Artifacts saved to: {exp_dir}")


def generate_exp_id() -> str:
    """Generate unique experiment ID based on timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp


class ExperimentTracker:
    """
    Track experiments with automatic logging and artifact saving.
    """
    
    def __init__(self, labeled_df: pd.DataFrame):
        self.labeled_df = labeled_df
        self.public_proxy, self.private_proxy = create_proxy_split(labeled_df)
        self.experiments = []
    
    def evaluate_oof(self, oof_df: pd.DataFrame, pred_col: str = 'prediction') -> Dict:
        """
        Evaluate OOF predictions on full set and proxy splits.
        """
        # Full CV evaluation
        cv_result = metric.evaluate(
            oof_df['is_cheating'].values,
            oof_df[pred_col].values,
            n_thresholds=N_THRESHOLDS
        )
        
        # Proxy evaluation
        proxy_results = evaluate_on_proxy(
            oof_df, self.public_proxy, self.private_proxy, pred_col
        )
        
        return {
            'cv_cost': cv_result.best_cost,
            'cv_score': cv_result.best_score,
            'cv_t_low': cv_result.t_low,
            'cv_t_high': cv_result.t_high,
            **proxy_results
        }
    
    def log_and_save(
        self,
        exp_name: str,
        oof_df: pd.DataFrame,
        cv_summary: Dict,
        pred_col: str = 'prediction',
        notes: str = "",
        model_config: Dict = None
    ) -> str:
        """
        Evaluate, log, and save experiment.
        Returns experiment ID.
        """
        exp_id = generate_exp_id()
        
        # Evaluate
        results = self.evaluate_oof(oof_df, pred_col)
        
        # Log
        log_experiment(
            exp_id=exp_id,
            exp_name=exp_name,
            cv_cost=results['cv_cost'],
            public_cost=results['public_cost'],
            private_cost=results['private_cost'],
            cv_t_low=results['cv_t_low'],
            cv_t_high=results['cv_t_high'],
            notes=notes,
        )
        
        # Save artifacts
        save_experiment_artifacts(exp_id, oof_df, cv_summary, model_config)
        
        self.experiments.append({
            'exp_id': exp_id,
            'exp_name': exp_name,
            **results
        })
        
        return exp_id
    
    def get_best_experiment(self) -> Dict:
        """Get the experiment with lowest public cost."""
        if not self.experiments:
            return None
        return min(self.experiments, key=lambda x: x['public_cost'])


if __name__ == "__main__":
    # Test experiment framework
    print("=" * 60)
    print("EXPERIMENT FRAMEWORK TEST")
    print("=" * 60)
    
    # Load data
    train = pd.read_csv("train.csv")
    labeled = train[train['is_cheating'].notna()].copy()
    labeled['is_cheating'] = labeled['is_cheating'].astype(int)
    
    print(f"Labeled samples: {len(labeled):,}")
    
    # Create tracker
    tracker = ExperimentTracker(labeled)
    
    # Create dummy OOF predictions
    np.random.seed(42)
    oof_df = labeled[['user_hash', 'is_cheating']].copy()
    oof_df['prediction'] = np.random.rand(len(oof_df))
    
    # Evaluate
    results = tracker.evaluate_oof(oof_df)
    
    print(f"\nDummy experiment results:")
    print(f"  CV Cost: {results['cv_cost']:,}")
    print(f"  Public Cost: {results['public_cost']:,}")
    print(f"  Private Cost: {results['private_cost']:,}")
