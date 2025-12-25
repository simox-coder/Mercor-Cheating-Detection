#!/usr/bin/env python3
"""
Run All Branches - Orchestrator Script
=======================================
Runs all 4 branches (A, B, C, D) and produces summary.

Usage:
    python run_all_branches.py                    # Run all branches
    python run_all_branches.py --branch A         # Run specific branch
    python run_all_branches.py --branch A B       # Run multiple branches
    python run_all_branches.py --quick            # Quick mode (fewer iterations)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import hashlib

from data_loader import set_seed, SEED


def compute_metric_hash():
    """Compute SHA256 of metric code block."""
    code_block = '''
COST_FN = 5000
COST_FP = 1000
COST_REVIEW = 50
pass_mask = y_pred < t_low
review_mask = (y_pred >= t_low) & (y_pred < t_high)
block_mask = y_pred >= t_high
n_fn = int(np.sum((y_true == 1) & pass_mask))
n_fp = int(np.sum((y_true == 0) & block_mask))
cost_fn_total = n_fn * COST_FN
cost_fp_total = n_fp * COST_FP
cost_review_total = n_review * COST_REVIEW
total_cost = cost_fn_total + cost_fp_total + cost_review_total
'''
    return hashlib.sha256(code_block.encode()).hexdigest()


def run_baseline_sanity():
    """Run baseline sanity checks."""
    import numpy as np
    import pandas as pd
    from data_loader import load_train_data, split_labeled_unlabeled, create_proxy_split
    import metric
    
    print("\n" + "="*60)
    print("BASELINE SANITY CHECKS")
    print("="*60)
    
    train = load_train_data()
    labeled, _ = split_labeled_unlabeled(train)
    public_proxy, _ = create_proxy_split(labeled, seed=SEED)
    
    y_true = public_proxy['is_cheating'].values
    n = len(y_true)
    
    baselines = [
        ('constant_0.01', np.full(n, 0.01)),
        ('constant_0.50', np.full(n, 0.50)),
        ('constant_0.99', np.full(n, 0.99)),
        ('random_uniform', np.random.RandomState(SEED).random(n))
    ]
    
    print(f"\nPublic proxy size: {n:,}")
    print(f"Cheater rate: {y_true.mean():.4f}")
    print("\nBaseline results:")
    print("-" * 50)
    
    for name, preds in baselines:
        result = metric.score(y_true, preds)
        print(f"  {name:20s}: cost={result['best_cost']:>12,}  "
              f"(t_low={result['t_low']:.2f}, t_high={result['t_high']:.2f})")
    
    print("-" * 50)
    print("Expected: no-signal baselines ~ 4-6M range")
    print("Good models should target ~ 1-2M range")
    print("TOP 1 target: ~ 1.54M")
    print("="*60)


def print_summary_table(results: dict):
    """Print final summary table."""
    print("\n" + "="*80)
    print("FINAL SUMMARY TABLE")
    print("="*80)
    print(f"{'Branch':<10} {'Method':<35} {'Proxy Public':>14} {'Proxy Private':>14}")
    print("-"*80)
    
    for branch, res in sorted(results.items()):
        method = res.get('method', 'Unknown')[:35]
        pub_cost = res.get('public_proxy_cost', 'N/A')
        priv_cost = res.get('private_proxy_cost', 'N/A')
        
        if isinstance(pub_cost, (int, float)):
            pub_str = f"{pub_cost:>14,}"
        else:
            pub_str = f"{pub_cost:>14}"
            
        if isinstance(priv_cost, (int, float)):
            priv_str = f"{priv_cost:>14,}"
        else:
            priv_str = f"{priv_cost:>14}"
        
        print(f"{branch:<10} {method:<35} {pub_str} {priv_str}")
    
    print("-"*80)
    print("\nSubmission files:")
    for branch, res in sorted(results.items()):
        sub_file = res.get('submission_file', f'submission_{branch}.csv')
        print(f"  {branch}: {sub_file}")
    
    print("\nThresholds (from OOF):")
    for branch, res in sorted(results.items()):
        t_low = res.get('t_low', 0)
        t_high = res.get('t_high', 1)
        print(f"  {branch}: t_low={t_low:.3f}, t_high={t_high:.3f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Run all branches for Mercor Cheating Detection')
    parser.add_argument('--branch', nargs='+', choices=['A', 'B', 'C', 'D', 'all'],
                        default=['all'], help='Branches to run')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer iterations')
    parser.add_argument('--skip-sanity', action='store_true', help='Skip baseline sanity checks')
    args = parser.parse_args()
    
    # Set global seed
    set_seed(SEED)
    
    # Determine branches to run
    if 'all' in args.branch:
        branches = ['A', 'B', 'C', 'D']
    else:
        branches = args.branch
    
    print("="*80)
    print("MERCOR CHEATING DETECTION PIPELINE")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print(f"Metric SHA256: {compute_metric_hash()}")
    print(f"Branches to run: {branches}")
    print(f"Quick mode: {args.quick}")
    print("="*80)
    
    # Run metric tests first
    print("\n--- Running metric parity tests ---")
    import metric_tests
    if not metric_tests.run_all_tests():
        print("METRIC TESTS FAILED - STOPPING PIPELINE")
        sys.exit(1)
    
    # Baseline sanity checks
    if not args.skip_sanity:
        run_baseline_sanity()
    
    # Run branches
    results = {}
    
    if 'A' in branches:
        print("\n" + "#"*80)
        print("# RUNNING BRANCH A")
        print("#"*80)
        from branch_a_tabular import run_branch_a
        try:
            results['A'] = run_branch_a(
                output_dir='artifacts/branch_A_tabular',
                verbose=True,
                quick_mode=args.quick
            )
        except Exception as e:
            print(f"Branch A failed: {e}")
            results['A'] = {'error': str(e)}
    
    if 'B' in branches:
        print("\n" + "#"*80)
        print("# RUNNING BRANCH B")
        print("#"*80)
        from branch_b_node2vec import run_branch_b
        try:
            results['B'] = run_branch_b(
                output_dir='artifacts/branch_B_node2vec',
                verbose=True,
                quick_mode=args.quick
            )
        except Exception as e:
            print(f"Branch B failed: {e}")
            results['B'] = {'error': str(e)}
    
    if 'C' in branches:
        print("\n" + "#"*80)
        print("# RUNNING BRANCH C")
        print("#"*80)
        from branch_c_labelprop import run_branch_c
        try:
            results['C'] = run_branch_c(
                output_dir='artifacts/branch_C_labelprop',
                verbose=True,
                quick_mode=args.quick
            )
        except Exception as e:
            print(f"Branch C failed: {e}")
            results['C'] = {'error': str(e)}
    
    if 'D' in branches:
        print("\n" + "#"*80)
        print("# RUNNING BRANCH D")
        print("#"*80)
        from branch_d_gnn import run_branch_d
        try:
            results['D'] = run_branch_d(
                output_dir='artifacts/branch_D_gnn',
                verbose=True,
                quick_mode=args.quick
            )
        except Exception as e:
            print(f"Branch D failed: {e}")
            results['D'] = {'error': str(e)}
    
    # Print summary
    print_summary_table(results)
    
    # Save combined results
    with open('artifacts/all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll results saved to: artifacts/all_results.json")
    print("\nDone!")
    
    return results


if __name__ == '__main__':
    main()
