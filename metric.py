"""
Metric wrapper that calls official_eval_reference.py evaluation logic.
Provides convenient interface for training pipelines.
"""

import numpy as np
from typing import Tuple, Optional
from official_eval_reference import (
    compute_cost,
    find_optimal_thresholds,
    evaluate,
    MISS_CHEATER_COST,
    BLOCK_CLEAN_COST,
    MANUAL_REVIEW_COST
)


def score(y_true: np.ndarray, y_pred: np.ndarray, n_thresholds: int = 100) -> float:
    """
    Compute the cost score (lower is better).
    
    Args:
        y_true: Binary ground truth
        y_pred: Predicted probabilities
        n_thresholds: Number of threshold candidates
        
    Returns:
        Total cost (float) - the main metric to minimize
    """
    result = evaluate(y_true, y_pred, n_thresholds)
    return result['total_cost']


def score_with_details(y_true: np.ndarray, y_pred: np.ndarray, 
                       n_thresholds: int = 100) -> Tuple[float, dict]:
    """
    Compute score with full details.
    
    Returns:
        Tuple of (total_cost, details_dict)
    """
    result = evaluate(y_true, y_pred, n_thresholds)
    return result['total_cost'], result


def score_at_thresholds(y_true: np.ndarray, y_pred: np.ndarray,
                        t_low: float, t_high: float) -> float:
    """
    Compute cost at specific thresholds (no optimization).
    
    Args:
        y_true: Binary ground truth
        y_pred: Predicted probabilities
        t_low: Lower threshold
        t_high: Upper threshold
        
    Returns:
        Total cost at these thresholds
    """
    result = compute_cost(y_true, y_pred, t_low, t_high)
    return result['total_cost']


def get_thresholds(y_true: np.ndarray, y_pred: np.ndarray,
                   n_thresholds: int = 100) -> Tuple[float, float]:
    """
    Find optimal thresholds only.
    
    Returns:
        Tuple of (t_low, t_high)
    """
    t_low, t_high, _ = find_optimal_thresholds(y_true, y_pred, n_thresholds)
    return t_low, t_high


# Negated score for maximization (e.g., with Optuna direction='maximize')
def neg_score(y_true: np.ndarray, y_pred: np.ndarray, n_thresholds: int = 100) -> float:
    """Returns negative cost (higher is better for maximization)."""
    return -score(y_true, y_pred, n_thresholds)


if __name__ == "__main__":
    # Test wrapper
    np.random.seed(42)
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([0.9, 0.8, 0.3, 0.1, 0.2, 0.6, 0.7, 0.05, 0.95, 0.4])
    
    cost = score(y_true, y_pred)
    print(f"Total cost: {cost}")
    
    cost2, details = score_with_details(y_true, y_pred)
    print(f"Details: {details}")
