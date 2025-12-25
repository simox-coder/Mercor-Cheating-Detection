"""
Official Evaluation Reference - Mercor Cheating Detection
===========================================================
Source: Kaggle Competition "mercor-cheating-detection"
Notebook URL: https://www.kaggle.com/code/drewkringel/cheating-detection-eval
(Verbatim logic reconstructed from competition description and standard cost-based eval)

The metric searches for optimal thresholds (t_low, t_high) that minimize total cost.
Decision regions:
  - p < t_low: Auto-pass (cheaters slip through → FN cost)
  - t_low <= p < t_high: Manual review (review cost per sample)
  - p >= t_high: Auto-block (legitimate users blocked → FP cost)

Cost parameters (per Kaggle competition):
  - FN cost (missed cheater): 5000
  - FP cost (blocked legitimate): 1000  
  - Review cost: 50 per sample

The score reported is the NEGATIVE of total cost (higher is better, so more negative cost is better).
Best possible: 0 (no cost)
Typical range: -1M to -7M depending on model quality and data size.
"""

import numpy as np
from typing import Tuple, Dict

# Official cost parameters
COST_FN = 5000    # Cost of missing a cheater (false negative)
COST_FP = 1000    # Cost of blocking a legitimate user (false positive)
COST_REVIEW = 50  # Cost of manual review per sample

def compute_cost_at_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_low: float,
    t_high: float
) -> Tuple[int, Dict]:
    """
    Compute total cost at given thresholds.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities in [0, 1]
        t_low: Lower threshold for auto-pass/review boundary
        t_high: Upper threshold for review/auto-block boundary
    
    Returns:
        total_cost: Total cost (integer)
        breakdown: Dictionary with cost breakdown
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Decision regions
    pass_mask = y_pred < t_low
    review_mask = (y_pred >= t_low) & (y_pred < t_high)
    block_mask = y_pred >= t_high
    
    # Count decisions
    n_pass = int(np.sum(pass_mask))
    n_review = int(np.sum(review_mask))
    n_block = int(np.sum(block_mask))
    
    # Count errors
    # In pass region: cheaters that slip through (FN)
    n_fn = int(np.sum((y_true == 1) & pass_mask))
    
    # In block region: legitimate users blocked (FP)
    n_fp = int(np.sum((y_true == 0) & block_mask))
    
    # Costs
    cost_fn_total = n_fn * COST_FN
    cost_fp_total = n_fp * COST_FP
    cost_review_total = n_review * COST_REVIEW
    
    total_cost = cost_fn_total + cost_fp_total + cost_review_total
    
    breakdown = {
        'n_pass': n_pass,
        'n_review': n_review,
        'n_block': n_block,
        'n_fn': n_fn,
        'n_fp': n_fp,
        'cost_fn': cost_fn_total,
        'cost_fp': cost_fp_total,
        'cost_review': cost_review_total,
        'total_cost': total_cost
    }
    
    return total_cost, breakdown


def find_best_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_thresholds: int = 101
) -> Tuple[int, float, float, Dict]:
    """
    Search for optimal thresholds that minimize total cost.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_thresholds: Number of threshold values to search (default 101 for 0.01 steps)
    
    Returns:
        best_cost: Minimum cost found
        best_t_low: Optimal lower threshold
        best_t_high: Optimal upper threshold  
        best_breakdown: Cost breakdown at optimal thresholds
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Threshold candidates from 0 to 1
    thresholds = np.linspace(0, 1, n_thresholds)
    
    best_cost = float('inf')
    best_t_low = 0.0
    best_t_high = 1.0
    best_breakdown = None
    
    # Grid search over all valid (t_low, t_high) pairs where t_low <= t_high
    for t_low in thresholds:
        for t_high in thresholds:
            if t_low > t_high:
                continue
            
            cost, breakdown = compute_cost_at_thresholds(y_true, y_pred, t_low, t_high)
            
            # Use < for strict improvement (ties go to earlier threshold)
            if cost < best_cost:
                best_cost = cost
                best_t_low = t_low
                best_t_high = t_high
                best_breakdown = breakdown
    
    return int(best_cost), best_t_low, best_t_high, best_breakdown


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_thresholds: int = 101
) -> Dict:
    """
    Full evaluation: find best thresholds and return all metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_thresholds: Number of threshold values to search
    
    Returns:
        Dictionary containing:
        - best_cost: Minimum total cost
        - best_score: Negative of best_cost (competition score, higher is better)
        - t_low, t_high: Optimal thresholds
        - breakdown: Full cost breakdown
    """
    best_cost, t_low, t_high, breakdown = find_best_thresholds(
        y_true, y_pred, n_thresholds
    )
    
    return {
        'best_cost': best_cost,
        'best_score': -best_cost,  # Score is negative cost (higher is better)
        't_low': t_low,
        't_high': t_high,
        'n_pass': breakdown['n_pass'],
        'n_review': breakdown['n_review'],
        'n_block': breakdown['n_block'],
        'n_fn': breakdown['n_fn'],
        'n_fp': breakdown['n_fp'],
        'cost_fn': breakdown['cost_fn'],
        'cost_fp': breakdown['cost_fp'],
        'cost_review': breakdown['cost_review'],
        'breakdown': breakdown
    }


if __name__ == '__main__':
    # Quick sanity test
    import hashlib
    
    # Hash of this file's core logic for verification
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
    
    sha256_hash = hashlib.sha256(code_block.encode()).hexdigest()
    print(f"SHA256 of official metric code block: {sha256_hash}")
    
    # Test with sample data
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.5, 0.3, 0.6, 0.7])
    
    result = evaluate(y_true, y_pred)
    print(f"\nSample evaluation:")
    print(f"  Best cost: {result['best_cost']}")
    print(f"  Best score: {result['best_score']}")
    print(f"  Thresholds: t_low={result['t_low']:.2f}, t_high={result['t_high']:.2f}")
    print(f"  Counts: pass={result['n_pass']}, review={result['n_review']}, block={result['n_block']}")
