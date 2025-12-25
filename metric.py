"""
Metric Wrapper Module
=====================
Thin wrapper around official_eval_reference.py for easy import and use.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import official_eval_reference as official

# Re-export cost constants
COST_FN = official.COST_FN
COST_FP = official.COST_FP  
COST_REVIEW = official.COST_REVIEW


def score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_thresholds: int = 101
) -> Dict:
    """
    Evaluate predictions and return full results.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities in [0, 1]
        n_thresholds: Number of thresholds to search (default 101)
    
    Returns:
        Dictionary containing:
        - best_cost: Minimum total cost (positive integer)
        - best_score: Negative of best_cost (competition metric, higher=better)
        - t_low: Optimal lower threshold
        - t_high: Optimal upper threshold
        - n_pass, n_review, n_block: Decision region counts
        - n_fn, n_fp: Error counts
        - cost_fn, cost_fp, cost_review: Cost components
        - breakdown: Full breakdown dict
    """
    return official.evaluate(y_true, y_pred, n_thresholds)


def compute_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_low: float,
    t_high: float
) -> Tuple[int, Dict]:
    """
    Compute cost at specific thresholds.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        t_low: Lower threshold
        t_high: Upper threshold
    
    Returns:
        total_cost: Total cost at these thresholds
        breakdown: Dictionary with cost breakdown
    """
    return official.compute_cost_at_thresholds(y_true, y_pred, t_low, t_high)


def find_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_thresholds: int = 101
) -> Tuple[float, float]:
    """
    Find optimal thresholds only.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_thresholds: Number of thresholds to search
    
    Returns:
        t_low: Optimal lower threshold
        t_high: Optimal upper threshold
    """
    _, t_low, t_high, _ = official.find_best_thresholds(y_true, y_pred, n_thresholds)
    return t_low, t_high


def best_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_thresholds: int = 101
) -> int:
    """
    Get best cost only.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_thresholds: Number of thresholds to search
    
    Returns:
        best_cost: Minimum total cost achievable
    """
    cost, _, _, _ = official.find_best_thresholds(y_true, y_pred, n_thresholds)
    return cost


def best_score_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_thresholds: int = 101
) -> int:
    """
    Get best score only (negative of best cost).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_thresholds: Number of thresholds to search
    
    Returns:
        best_score: -best_cost (higher is better)
    """
    return -best_cost(y_true, y_pred, n_thresholds)


def evaluate_at_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_low: float,
    t_high: float
) -> Dict:
    """
    Full evaluation at specific thresholds.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        t_low: Lower threshold
        t_high: Upper threshold
    
    Returns:
        Dictionary with cost, score, counts, breakdown
    """
    cost, breakdown = official.compute_cost_at_thresholds(y_true, y_pred, t_low, t_high)
    return {
        'cost': cost,
        'score': -cost,
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


def print_evaluation(result: Dict) -> None:
    """Pretty print evaluation results."""
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║           EVALUATION RESULTS                     ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║ Best Score: {result['best_score']:>14,}                    ║")
    print(f"║ Best Cost:  {result['best_cost']:>14,}                    ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║ Thresholds: t_low={result['t_low']:.3f}, t_high={result['t_high']:.3f}       ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║ Decision Counts:                                 ║")
    print(f"║   Auto-pass:   {result['n_pass']:>8,}                       ║")
    print(f"║   Review:      {result['n_review']:>8,}                       ║")
    print(f"║   Auto-block:  {result['n_block']:>8,}                       ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║ Cost Breakdown:                                  ║")
    print(f"║   FN Cost ({result['n_fn']:>5} cheaters missed):    {result['cost_fn']:>10,} ║")
    print(f"║   FP Cost ({result['n_fp']:>5} legitim blocked):    {result['cost_fp']:>10,} ║")
    print(f"║   Review Cost:                     {result['cost_review']:>10,} ║")
    print(f"╚══════════════════════════════════════════════════╝")


if __name__ == '__main__':
    # Test the wrapper
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.5, 0.3, 0.6, 0.7])
    
    result = score(y_true, y_pred)
    print_evaluation(result)
