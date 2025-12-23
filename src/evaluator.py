"""Local cost metric evaluator matching the competition metric."""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy.optimize import minimize_scalar

from .config import COST_FN, COST_REVIEW, COST_FP


def compute_cost(y_true: np.ndarray, y_pred: np.ndarray, 
                 low_thresh: float, high_thresh: float) -> float:
    """
    Compute total cost given predictions and thresholds.
    
    Regions:
    - auto-pass (pred < low_thresh): cost = 500 if cheater, 0 if clean
    - manual review (low_thresh <= pred < high_thresh): cost = 25 for all
    - auto-block (pred >= high_thresh): cost = 0 if cheater, 100 if clean
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Region masks
    auto_pass = y_pred < low_thresh
    manual_review = (y_pred >= low_thresh) & (y_pred < high_thresh)
    auto_block = y_pred >= high_thresh
    
    # Compute costs
    cost = 0.0
    
    # Auto-pass costs: 500 for each cheater
    cost += COST_FN * (auto_pass & (y_true == 1)).sum()
    
    # Manual review costs: 25 for each user
    cost += COST_REVIEW * manual_review.sum()
    
    # Auto-block costs: 100 for each clean user
    cost += COST_FP * (auto_block & (y_true == 0)).sum()
    
    return cost


def find_optimal_thresholds(y_true: np.ndarray, y_pred: np.ndarray,
                            n_search: int = 100) -> Tuple[float, float, float]:
    """
    Search for optimal low/high thresholds to minimize cost.
    
    Returns: (best_low_thresh, best_high_thresh, best_cost)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Get unique prediction values as candidate thresholds
    unique_preds = np.unique(y_pred)
    
    # If too many unique values, sample
    if len(unique_preds) > n_search:
        # Use quantiles for threshold candidates
        thresholds = np.percentile(y_pred, np.linspace(0, 100, n_search))
        thresholds = np.unique(thresholds)
    else:
        thresholds = unique_preds
    
    # Add boundary values
    thresholds = np.concatenate([[0.0], thresholds, [1.0]])
    thresholds = np.unique(thresholds)
    
    best_cost = float('inf')
    best_low = 0.0
    best_high = 1.0
    
    # Grid search over threshold pairs
    for low_thresh in thresholds:
        for high_thresh in thresholds:
            if high_thresh <= low_thresh:
                continue
            
            cost = compute_cost(y_true, y_pred, low_thresh, high_thresh)
            
            if cost < best_cost:
                best_cost = cost
                best_low = low_thresh
                best_high = high_thresh
    
    return best_low, best_high, best_cost


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         return_thresholds: bool = False) -> Dict:
    """
    Evaluate predictions using the competition cost metric.
    
    Returns dictionary with:
    - total_cost: Total cost (lower is better)
    - cost_score: -total_cost (higher is better, matches Kaggle)
    - low_thresh, high_thresh: Optimal thresholds
    - breakdown: Cost breakdown by region
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Find optimal thresholds
    low_thresh, high_thresh, total_cost = find_optimal_thresholds(y_true, y_pred)
    
    # Compute breakdown
    auto_pass = y_pred < low_thresh
    manual_review = (y_pred >= low_thresh) & (y_pred < high_thresh)
    auto_block = y_pred >= high_thresh
    
    fn_cost = COST_FN * (auto_pass & (y_true == 1)).sum()
    review_cost = COST_REVIEW * manual_review.sum()
    fp_cost = COST_FP * (auto_block & (y_true == 0)).sum()
    
    result = {
        "total_cost": total_cost,
        "cost_score": -total_cost,  # Higher is better (Kaggle format)
        "low_thresh": low_thresh,
        "high_thresh": high_thresh,
        "breakdown": {
            "false_negative_cost": fn_cost,
            "review_cost": review_cost,
            "false_positive_cost": fp_cost,
            "n_auto_pass": auto_pass.sum(),
            "n_review": manual_review.sum(),
            "n_auto_block": auto_block.sum(),
        }
    }
    
    return result


def evaluate_oof(oof_df: pd.DataFrame, label_col: str = "is_cheating",
                 pred_col: str = "prediction") -> Dict:
    """Evaluate OOF predictions from a DataFrame."""
    y_true = oof_df[label_col].values
    y_pred = oof_df[pred_col].values
    return evaluate_predictions(y_true, y_pred)


def print_evaluation(result: Dict):
    """Pretty print evaluation results."""
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Cost: {result['total_cost']:,.0f}")
    print(f"Cost Score (Kaggle format): {result['cost_score']:,.0f}")
    print(f"Optimal Thresholds: low={result['low_thresh']:.4f}, high={result['high_thresh']:.4f}")
    print()
    print("Cost Breakdown:")
    bd = result['breakdown']
    print(f"  - False Negative Cost (auto-pass cheaters): {bd['false_negative_cost']:,.0f}")
    print(f"  - Review Cost: {bd['review_cost']:,.0f}")
    print(f"  - False Positive Cost (auto-block clean): {bd['false_positive_cost']:,.0f}")
    print()
    print("Decision Distribution:")
    print(f"  - Auto-pass: {bd['n_auto_pass']:,}")
    print(f"  - Manual review: {bd['n_review']:,}")
    print(f"  - Auto-block: {bd['n_auto_block']:,}")
    print("=" * 50)


if __name__ == "__main__":
    # Test the evaluator
    np.random.seed(42)
    
    # Generate test data
    n = 10000
    y_true = np.random.binomial(1, 0.3, n)
    y_pred = y_true * 0.5 + np.random.uniform(0, 0.5, n)
    y_pred = np.clip(y_pred, 0, 1)
    
    result = evaluate_predictions(y_true, y_pred)
    print_evaluation(result)
