"""
Official Kaggle evaluation code for Mercor Cheating Detection competition.
Reference: https://www.kaggle.com/code/drewkringel/cheating-detection-eval

The metric is a cost-based evaluation with 3 regions:
- Auto-pass (prediction < t_low): No manual review needed
  - Cost for actual cheater: miss_cheater_cost (high penalty for missing a cheater)
  - Cost for actual clean: 0 (correct decision)
- Manual review (t_low <= prediction < t_high): Manual review required
  - Cost: manual_review_cost per sample (moderate cost for human review)
- Auto-block (prediction >= t_high): Automatically blocked
  - Cost for actual cheater: 0 (correct decision)
  - Cost for actual clean: block_clean_cost (penalty for blocking clean user)

The evaluator searches for optimal (t_low, t_high) thresholds to minimize total cost.
"""

import numpy as np
from typing import Tuple

# Cost parameters (from Kaggle competition)
MISS_CHEATER_COST = 500.0      # Cost of auto-passing a cheater
BLOCK_CLEAN_COST = 100.0       # Cost of auto-blocking a clean user
MANUAL_REVIEW_COST = 50.0      # Cost of manual review

def compute_cost(y_true: np.ndarray, y_pred: np.ndarray, t_low: float, t_high: float) -> dict:
    """
    Compute cost given thresholds.
    
    Args:
        y_true: Binary ground truth (1 = cheater, 0 = clean)
        y_pred: Predicted probabilities [0, 1]
        t_low: Lower threshold for auto-pass region
        t_high: Upper threshold for auto-block region
        
    Returns:
        Dictionary with cost breakdown and region counts
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Define regions
    auto_pass = y_pred < t_low                     # Region 1: Auto-pass
    manual_review = (y_pred >= t_low) & (y_pred < t_high)  # Region 2: Manual review
    auto_block = y_pred >= t_high                  # Region 3: Auto-block
    
    # Counts per region
    n_auto_pass = auto_pass.sum()
    n_manual_review = manual_review.sum()
    n_auto_block = auto_block.sum()
    
    # Cost components
    # Auto-pass region: cost for missed cheaters
    missed_cheaters = ((y_true == 1) & auto_pass).sum()
    cost_missed = missed_cheaters * MISS_CHEATER_COST
    
    # Manual review region: cost for all samples
    cost_manual = n_manual_review * MANUAL_REVIEW_COST
    
    # Auto-block region: cost for blocked clean users
    blocked_clean = ((y_true == 0) & auto_block).sum()
    cost_block = blocked_clean * BLOCK_CLEAN_COST
    
    total_cost = cost_missed + cost_manual + cost_block
    
    return {
        'total_cost': total_cost,
        'cost_missed_cheaters': cost_missed,
        'cost_manual_review': cost_manual,
        'cost_blocked_clean': cost_block,
        'n_auto_pass': int(n_auto_pass),
        'n_manual_review': int(n_manual_review),
        'n_auto_block': int(n_auto_block),
        'missed_cheaters': int(missed_cheaters),
        'blocked_clean': int(blocked_clean),
        't_low': t_low,
        't_high': t_high
    }


def find_optimal_thresholds(y_true: np.ndarray, y_pred: np.ndarray, 
                           n_thresholds: int = 100) -> Tuple[float, float, dict]:
    """
    Find optimal (t_low, t_high) thresholds that minimize total cost.
    
    Args:
        y_true: Binary ground truth
        y_pred: Predicted probabilities
        n_thresholds: Number of threshold candidates to try
        
    Returns:
        Tuple of (best_t_low, best_t_high, best_result_dict)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Generate threshold candidates
    thresholds = np.linspace(0, 1, n_thresholds + 1)
    
    best_cost = float('inf')
    best_t_low = 0.0
    best_t_high = 1.0
    best_result = None
    
    # Grid search over t_low and t_high where t_low <= t_high
    for t_low in thresholds:
        for t_high in thresholds:
            if t_low > t_high:
                continue
            
            result = compute_cost(y_true, y_pred, t_low, t_high)
            
            if result['total_cost'] < best_cost:
                best_cost = result['total_cost']
                best_t_low = t_low
                best_t_high = t_high
                best_result = result
    
    return best_t_low, best_t_high, best_result


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_thresholds: int = 100) -> dict:
    """
    Main evaluation function - finds optimal thresholds and returns metrics.
    
    Args:
        y_true: Binary ground truth (1 = cheater, 0 = clean)
        y_pred: Predicted probabilities [0, 1]
        n_thresholds: Number of threshold candidates
        
    Returns:
        Dictionary with optimal thresholds and all cost metrics
    """
    t_low, t_high, result = find_optimal_thresholds(y_true, y_pred, n_thresholds)
    return result


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([0.9, 0.8, 0.3, 0.1, 0.2, 0.6, 0.7, 0.05, 0.95, 0.4])
    
    result = evaluate(y_true, y_pred)
    print("Evaluation result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
