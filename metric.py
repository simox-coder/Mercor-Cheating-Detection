"""
Metric wrapper for Mercor Cheating Detection Competition.

This module wraps the official evaluation logic from official_eval_reference.py
and provides convenient interfaces for CV evaluation and submission scoring.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import official_eval_reference as official


@dataclass
class MetricResult:
    """Container for evaluation results."""
    best_cost: int
    best_score: int  # = -best_cost (Kaggle score format)
    t_low: float
    t_high: float
    n_auto_pass: int
    n_manual_review: int
    n_auto_block: int
    breakdown: Dict[str, int]


def apply_thresholds(pred: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """
    Apply thresholds to predictions to get decision regions.
    
    Args:
        pred: Predicted probabilities in [0, 1]
        t_low: Lower threshold
        t_high: Upper threshold
        
    Returns:
        decisions: Array with values 'pass', 'review', or 'block'
    """
    decisions = np.empty(len(pred), dtype=object)
    decisions[pred < t_low] = 'pass'
    decisions[(pred >= t_low) & (pred < t_high)] = 'review'
    decisions[pred >= t_high] = 'block'
    return decisions


def validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Validate metric inputs.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Raises:
        AssertionError: If inputs are invalid
    """
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    assert np.all(np.isin(y_true, [0, 1])), "y_true must contain only 0 and 1"
    assert np.min(y_pred) >= 0, f"y_pred min {np.min(y_pred)} < 0"
    assert np.max(y_pred) <= 1, f"y_pred max {np.max(y_pred)} > 1"
    assert not np.any(np.isnan(y_pred)), "y_pred contains NaN values"


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, 
             n_thresholds: int = 100, validate: bool = True) -> MetricResult:
    """
    Evaluate predictions using the official competition metric.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities in [0, 1]
        n_thresholds: Number of threshold values to search
        validate: Whether to validate inputs
        
    Returns:
        MetricResult with best cost, score, thresholds, and breakdown
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if validate:
        validate_inputs(y_true, y_pred)
    
    best_cost, t_low, t_high, breakdown = official.find_optimal_thresholds(
        y_true, y_pred, n_thresholds
    )
    
    return MetricResult(
        best_cost=best_cost,
        best_score=-best_cost,
        t_low=t_low,
        t_high=t_high,
        n_auto_pass=breakdown['n_auto_pass'],
        n_manual_review=breakdown['n_manual_review'],
        n_auto_block=breakdown['n_auto_block'],
        breakdown=breakdown
    )


def compute_cost_at_thresholds(y_true: np.ndarray, y_pred: np.ndarray,
                               t_low: float, t_high: float) -> Tuple[int, Dict[str, int]]:
    """
    Compute cost at specific thresholds (useful for cross-validation).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        t_low: Lower threshold
        t_high: Upper threshold
        
    Returns:
        cost: Total cost
        breakdown: Cost breakdown dictionary
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return official.compute_cost_for_thresholds(y_true, y_pred, t_low, t_high)


def print_result(result: MetricResult, prefix: str = "") -> None:
    """Print evaluation result in a formatted way."""
    print(f"{prefix}Cost: {result.best_cost:,} | Score: {result.best_score:,}")
    print(f"{prefix}Thresholds: t_low={result.t_low:.4f}, t_high={result.t_high:.4f}")
    print(f"{prefix}Regions: auto-pass={result.n_auto_pass:,}, "
          f"manual={result.n_manual_review:,}, auto-block={result.n_auto_block:,}")
    print(f"{prefix}Breakdown:")
    for key, value in result.breakdown.items():
        if key != 'total_cost':
            print(f"{prefix}  {key}: {value:,}")


if __name__ == "__main__":
    # Test the metric wrapper
    import pandas as pd
    
    print("=" * 60)
    print("METRIC WRAPPER TEST")
    print("=" * 60)
    
    # Simple test case
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    
    result = evaluate(y_true, y_pred)
    print("\nSimple test case:")
    print_result(result)
    
    # Test apply_thresholds
    decisions = apply_thresholds(y_pred, result.t_low, result.t_high)
    print(f"\nDecisions at optimal thresholds: {decisions}")
    
    print("\n" + "=" * 60)
    print("VALIDATION TEST")
    print("=" * 60)
    
    # Test validation
    try:
        validate_inputs(y_true, np.array([0.1, 0.9, 0.2, 1.5]))  # out of range
        print("ERROR: Should have raised assertion")
    except AssertionError as e:
        print(f"Correctly caught invalid input: {e}")
