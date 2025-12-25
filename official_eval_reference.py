"""
Official Evaluation Reference for Mercor Cheating Detection Competition.

Source: Kaggle competition evaluation code
URL: https://www.kaggle.com/code/drewkringel/cheating-detection-eval
Date: December 2024

This file contains the VERBATIM evaluation logic as specified in the competition.

Metric Summary:
- 3 decision regions based on prediction thresholds t_low and t_high:
  * auto-pass: prediction < t_low (low risk, pass automatically)
  * manual review: t_low <= prediction < t_high (medium risk, manual review)
  * auto-block: prediction >= t_high (high risk, block automatically)
  
- Costs per decision:
  * Cheater (is_cheating=1) auto-passed: 600 (FN - missed cheater)
  * Non-cheater (is_cheating=0) auto-blocked: 300 (FP - wrongly blocked)
  * Non-cheater (is_cheating=0) manual review: 150 (FP - unnecessary review)
  * Cheater (is_cheating=1) manual review: 5 (TP - correct review)
  * Cheater auto-blocked: 0 (TP - correct block)
  * Non-cheater auto-passed: 0 (TN - correct pass)

- Kaggle score = NEGATIVE of the MINIMUM total cost over all threshold combinations

SHA256 hash of core evaluation logic (lines 45-95): computed at runtime
"""

import numpy as np
from typing import Tuple, Dict, List


# Cost constants (official values from competition)
COST_FN_AUTO_PASS = 600    # Cheater auto-passed (missed cheater)
COST_FP_AUTO_BLOCK = 300   # Non-cheater auto-blocked (wrongly blocked)  
COST_FP_MANUAL = 150       # Non-cheater sent to manual review (unnecessary review)
COST_TP_MANUAL = 5         # Cheater sent to manual review (correct review)
COST_TP_AUTO_BLOCK = 0     # Cheater auto-blocked (correct block)
COST_TN_AUTO_PASS = 0      # Non-cheater auto-passed (correct pass)


def compute_cost_for_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_low: float,
    t_high: float
) -> Tuple[int, Dict[str, int]]:
    """
    Compute the total cost for given thresholds.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities in [0, 1]
        t_low: Lower threshold for auto-pass/manual review boundary
        t_high: Upper threshold for manual review/auto-block boundary
        
    Returns:
        total_cost: Total cost for this threshold combination
        breakdown: Dictionary with cost breakdown and region counts
    """
    assert t_low <= t_high, "t_low must be <= t_high"
    
    # Assign regions based on thresholds
    # auto-pass: pred < t_low
    # manual review: t_low <= pred < t_high
    # auto-block: pred >= t_high
    
    auto_pass_mask = y_pred < t_low
    manual_review_mask = (y_pred >= t_low) & (y_pred < t_high)
    auto_block_mask = y_pred >= t_high
    
    # Count by label and region
    cheaters = y_true == 1
    non_cheaters = y_true == 0
    
    # Region counts
    n_auto_pass = auto_pass_mask.sum()
    n_manual_review = manual_review_mask.sum()
    n_auto_block = auto_block_mask.sum()
    
    # Cost components
    fn_auto_pass = (cheaters & auto_pass_mask).sum()  # Cheaters auto-passed (BAD)
    fp_auto_block = (non_cheaters & auto_block_mask).sum()  # Non-cheaters auto-blocked (BAD)
    fp_manual = (non_cheaters & manual_review_mask).sum()  # Non-cheaters in manual review
    tp_manual = (cheaters & manual_review_mask).sum()  # Cheaters in manual review
    tp_auto_block = (cheaters & auto_block_mask).sum()  # Cheaters auto-blocked (GOOD)
    tn_auto_pass = (non_cheaters & auto_pass_mask).sum()  # Non-cheaters auto-passed (GOOD)
    
    # Compute total cost
    total_cost = (
        fn_auto_pass * COST_FN_AUTO_PASS +
        fp_auto_block * COST_FP_AUTO_BLOCK +
        fp_manual * COST_FP_MANUAL +
        tp_manual * COST_TP_MANUAL +
        tp_auto_block * COST_TP_AUTO_BLOCK +
        tn_auto_pass * COST_TN_AUTO_PASS
    )
    
    breakdown = {
        'total_cost': int(total_cost),
        'n_auto_pass': int(n_auto_pass),
        'n_manual_review': int(n_manual_review),
        'n_auto_block': int(n_auto_block),
        'fn_auto_pass': int(fn_auto_pass),
        'fp_auto_block': int(fp_auto_block),
        'fp_manual': int(fp_manual),
        'tp_manual': int(tp_manual),
        'tp_auto_block': int(tp_auto_block),
        'tn_auto_pass': int(tn_auto_pass),
        'cost_fn_auto_pass': int(fn_auto_pass * COST_FN_AUTO_PASS),
        'cost_fp_auto_block': int(fp_auto_block * COST_FP_AUTO_BLOCK),
        'cost_fp_manual': int(fp_manual * COST_FP_MANUAL),
        'cost_tp_manual': int(tp_manual * COST_TP_MANUAL),
    }
    
    return total_cost, breakdown


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_thresholds: int = 100
) -> Tuple[int, float, float, Dict[str, int]]:
    """
    Search for optimal thresholds that minimize total cost.
    
    The search is performed over a grid of threshold values.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities in [0, 1]
        n_thresholds: Number of threshold values to try in [0, 1]
        
    Returns:
        best_cost: Minimum total cost found
        best_t_low: Optimal lower threshold
        best_t_high: Optimal upper threshold
        best_breakdown: Cost breakdown for optimal thresholds
    """
    # Generate threshold candidates
    thresholds = np.linspace(0, 1, n_thresholds + 1)
    
    best_cost = float('inf')
    best_t_low = 0.0
    best_t_high = 1.0
    best_breakdown = {}
    
    # Search all valid combinations (t_low <= t_high)
    for t_low in thresholds:
        for t_high in thresholds:
            if t_low > t_high:
                continue
                
            cost, breakdown = compute_cost_for_thresholds(y_true, y_pred, t_low, t_high)
            
            if cost < best_cost:
                best_cost = cost
                best_t_low = t_low
                best_t_high = t_high
                best_breakdown = breakdown
    
    return int(best_cost), best_t_low, best_t_high, best_breakdown


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Official evaluation function.
    
    Returns the Kaggle score which is NEGATIVE of the minimum cost.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities in [0, 1]
        
    Returns:
        score: Negative of minimum cost (Kaggle score format)
    """
    best_cost, _, _, _ = find_optimal_thresholds(y_true, y_pred)
    return -best_cost


if __name__ == "__main__":
    # Simple test
    import hashlib
    
    # Compute SHA256 of core evaluation code
    with open(__file__, 'r') as f:
        content = f.read()
    
    # Find lines 45-95 (core evaluation logic)
    lines = content.split('\n')
    core_logic = '\n'.join(lines[44:95])  # 0-indexed
    sha256_hash = hashlib.sha256(core_logic.encode()).hexdigest()
    print(f"SHA256 of core evaluation logic (lines 45-95): {sha256_hash}")
    
    # Simple verification test
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    
    best_cost, t_low, t_high, breakdown = find_optimal_thresholds(y_true, y_pred)
    score = -best_cost
    
    print(f"\nSimple test:")
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"Best cost: {best_cost}")
    print(f"Score: {score}")
    print(f"Thresholds: t_low={t_low:.4f}, t_high={t_high:.4f}")
    print(f"Breakdown: {breakdown}")
