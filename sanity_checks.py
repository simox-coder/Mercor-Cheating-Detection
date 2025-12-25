"""
Baseline sanity checks and scale validation for Mercor Cheating Detection.

This script verifies:
- G2: Dataset integrity (labeled rows only for metric)
- G2.a: Public/private proxy split
- G3: Baseline scale sanity (constant predictions)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

import metric


def load_labeled_data(train_path: str = "train.csv"):
    """Load and filter to labeled rows only."""
    train = pd.read_csv(train_path)
    
    # Filter to labeled rows (is_cheating is not NA)
    labeled = train[train['is_cheating'].notna()].copy()
    labeled['is_cheating'] = labeled['is_cheating'].astype(int)
    
    print(f"Total train rows: {len(train):,}")
    print(f"Labeled rows: {len(labeled):,}")
    print(f"Unlabeled rows: {len(train) - len(labeled):,}")
    print(f"Cheaters: {(labeled['is_cheating'] == 1).sum():,} ({100*(labeled['is_cheating'] == 1).mean():.1f}%)")
    print(f"Non-cheaters: {(labeled['is_cheating'] == 0).sum():,}")
    
    return labeled


def create_proxy_split(labeled_df: pd.DataFrame, seed: int = 42):
    """
    Create public/private proxy split (50/50 stratified).
    
    Args:
        labeled_df: DataFrame with labeled rows
        seed: Random seed for reproducibility
        
    Returns:
        public_df, private_df: Split DataFrames
    """
    public_df, private_df = train_test_split(
        labeled_df,
        test_size=0.5,
        stratify=labeled_df['is_cheating'],
        random_state=seed
    )
    
    print(f"\nProxy split (SEED={seed}):")
    print(f"  Public proxy:  {len(public_df):,} rows")
    print(f"  Private proxy: {len(private_df):,} rows")
    print(f"  Public cheater rate:  {100*public_df['is_cheating'].mean():.1f}%")
    print(f"  Private cheater rate: {100*private_df['is_cheating'].mean():.1f}%")
    
    return public_df, private_df


def run_baseline_sanity(y_true: np.ndarray, name: str = ""):
    """
    Run baseline sanity checks with constant and random predictions.
    
    G3: If ALL costs >> 5,000,000, this indicates a bug.
    Expected scale for ~50k samples: ~1-2M range
    """
    print(f"\n{'='*60}")
    print(f"BASELINE SANITY CHECK: {name}")
    print(f"{'='*60}")
    print(f"n_samples: {len(y_true):,}")
    print(f"n_cheaters: {y_true.sum():,} ({100*y_true.mean():.1f}%)")
    print(f"n_non_cheaters: {(1-y_true).sum():,}")
    
    n = len(y_true)
    results = {}
    
    # Test 1: Constant 0.01 (predict all clean)
    y_pred = np.full(n, 0.01)
    result = metric.evaluate(y_true, y_pred)
    results['const_0.01'] = result
    print(f"\nConstant 0.01 (predict all clean):")
    print(f"  Cost: {result.best_cost:,} | Score: {result.best_score:,}")
    print(f"  Thresholds: t_low={result.t_low:.2f}, t_high={result.t_high:.2f}")
    print(f"  Regions: pass={result.n_auto_pass:,}, review={result.n_manual_review:,}, block={result.n_auto_block:,}")
    
    # Test 2: Constant 0.50 (uncertain)
    y_pred = np.full(n, 0.50)
    result = metric.evaluate(y_true, y_pred)
    results['const_0.50'] = result
    print(f"\nConstant 0.50 (uncertain):")
    print(f"  Cost: {result.best_cost:,} | Score: {result.best_score:,}")
    print(f"  Thresholds: t_low={result.t_low:.2f}, t_high={result.t_high:.2f}")
    print(f"  Regions: pass={result.n_auto_pass:,}, review={result.n_manual_review:,}, block={result.n_auto_block:,}")
    
    # Test 3: Constant 0.99 (predict all cheaters)
    y_pred = np.full(n, 0.99)
    result = metric.evaluate(y_true, y_pred)
    results['const_0.99'] = result
    print(f"\nConstant 0.99 (predict all cheaters):")
    print(f"  Cost: {result.best_cost:,} | Score: {result.best_score:,}")
    print(f"  Thresholds: t_low={result.t_low:.2f}, t_high={result.t_high:.2f}")
    print(f"  Regions: pass={result.n_auto_pass:,}, review={result.n_manual_review:,}, block={result.n_auto_block:,}")
    
    # Test 4: Uniform random (fixed seed)
    np.random.seed(42)
    y_pred = np.random.rand(n)
    result = metric.evaluate(y_true, y_pred)
    results['random'] = result
    print(f"\nUniform random (seed=42):")
    print(f"  Cost: {result.best_cost:,} | Score: {result.best_score:,}")
    print(f"  Thresholds: t_low={result.t_low:.2f}, t_high={result.t_high:.2f}")
    print(f"  Regions: pass={result.n_auto_pass:,}, review={result.n_manual_review:,}, block={result.n_auto_block:,}")
    
    # Scale sanity check
    all_costs = [r.best_cost for r in results.values()]
    max_cost = max(all_costs)
    min_cost = min(all_costs)
    
    print(f"\n{'='*60}")
    print("SCALE SANITY CHECK")
    print(f"{'='*60}")
    print(f"Cost range: [{min_cost:,}, {max_cost:,}]")
    
    # Check for the ~7M bug (should be ~1.5M for ~50k samples with good model)
    # For constant predictions, manual review all is optimal
    # Cost = n_non_cheaters * 150 + n_cheaters * 5
    n_cheaters = y_true.sum()
    n_non_cheaters = len(y_true) - n_cheaters
    expected_manual_all = n_non_cheaters * 150 + n_cheaters * 5
    print(f"Expected manual-review-all cost: {expected_manual_all:,}")
    
    # Constant predictions should yield manual-review-all cost
    # Scale is correct if costs match this expectation
    for key, result in results.items():
        if abs(result.best_cost - expected_manual_all) > 10:
            print(f"⚠️ WARNING: {key} cost {result.best_cost:,} != expected {expected_manual_all:,}")
            return False, results
    
    print("✓ Scale sanity PASSED (constant predictions match expected manual-review-all cost)")
    return True, results


def main():
    print("="*60)
    print("MERCOR CHEATING DETECTION - BASELINE SANITY CHECKS")
    print("="*60)
    
    # Load labeled data
    labeled = load_labeled_data("train.csv")
    y_true_full = labeled['is_cheating'].values
    
    # Create proxy split
    public_df, private_df = create_proxy_split(labeled, seed=42)
    
    # First 5 user_hash values (for verification)
    print(f"\nFirst 5 user_hash in public proxy:")
    for i, h in enumerate(public_df['user_hash'].head(5)):
        print(f"  {i+1}. {h}")
    
    print(f"\nFirst 5 user_hash in private proxy:")
    for i, h in enumerate(private_df['user_hash'].head(5)):
        print(f"  {i+1}. {h}")
    
    # Run sanity checks on public proxy
    y_true_public = public_df['is_cheating'].values
    y_true_private = private_df['is_cheating'].values
    
    passed_public, results_public = run_baseline_sanity(y_true_public, "PUBLIC PROXY")
    passed_private, results_private = run_baseline_sanity(y_true_private, "PRIVATE PROXY")
    
    # Compare public vs private
    print(f"\n{'='*60}")
    print("PUBLIC vs PRIVATE COMPARISON")
    print(f"{'='*60}")
    
    for key in results_public:
        pub_cost = results_public[key].best_cost
        priv_cost = results_private[key].best_cost
        diff_pct = 100 * (pub_cost - priv_cost) / priv_cost if priv_cost > 0 else 0
        print(f"{key:12s}: public={pub_cost:>10,} | private={priv_cost:>10,} | diff={diff_pct:+.1f}%")
    
    # Full dataset sanity
    passed_full, results_full = run_baseline_sanity(y_true_full, "FULL LABELED")
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Public proxy passed:  {'✓' if passed_public else '✗'}")
    print(f"Private proxy passed: {'✓' if passed_private else '✗'}")
    print(f"Full labeled passed:  {'✓' if passed_full else '✗'}")
    
    all_passed = passed_public and passed_private and passed_full
    
    if all_passed:
        print("\n✓ ALL SANITY CHECKS PASSED - Ready to proceed")
        return 0
    else:
        print("\n✗ SANITY CHECKS FAILED - DO NOT PROCEED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
