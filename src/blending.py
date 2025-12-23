"""Ensemble blending and optimization utilities."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.stats import rankdata

from .evaluator import evaluate_predictions, compute_cost, find_optimal_thresholds


def blend_predictions(pred_dict: Dict[str, np.ndarray], 
                      weights: Dict[str, float] = None) -> np.ndarray:
    """
    Blend predictions from multiple models.
    
    Args:
        pred_dict: Dict mapping model name to predictions array
        weights: Dict mapping model name to weight (default: equal weights)
    
    Returns:
        Blended predictions array
    """
    if weights is None:
        weights = {k: 1.0 / len(pred_dict) for k in pred_dict}
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    # Blend
    result = np.zeros_like(list(pred_dict.values())[0])
    for name, preds in pred_dict.items():
        result += weights.get(name, 0) * preds
    
    return result


def rank_blend(pred_dict: Dict[str, np.ndarray],
               weights: Dict[str, float] = None) -> np.ndarray:
    """
    Blend predictions using rank averaging.
    
    Args:
        pred_dict: Dict mapping model name to predictions array
        weights: Dict mapping model name to weight (default: equal weights)
    
    Returns:
        Blended predictions (normalized to [0, 1])
    """
    if weights is None:
        weights = {k: 1.0 / len(pred_dict) for k in pred_dict}
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    # Convert to ranks
    n = len(list(pred_dict.values())[0])
    rank_sum = np.zeros(n)
    
    for name, preds in pred_dict.items():
        ranks = rankdata(preds) / n  # Normalize to [0, 1]
        rank_sum += weights.get(name, 0) * ranks
    
    return rank_sum


def logit_blend(pred_dict: Dict[str, np.ndarray],
                weights: Dict[str, float] = None,
                eps: float = 1e-7) -> np.ndarray:
    """
    Blend predictions in logit space.
    
    Args:
        pred_dict: Dict mapping model name to predictions array
        weights: Dict mapping model name to weight (default: equal weights)
        eps: Small constant to avoid log(0)
    
    Returns:
        Blended predictions
    """
    if weights is None:
        weights = {k: 1.0 / len(pred_dict) for k in pred_dict}
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    # Convert to logits and blend
    logit_sum = np.zeros_like(list(pred_dict.values())[0])
    
    for name, preds in pred_dict.items():
        # Clip to avoid inf
        preds_clipped = np.clip(preds, eps, 1 - eps)
        logits = np.log(preds_clipped / (1 - preds_clipped))
        logit_sum += weights.get(name, 0) * logits
    
    # Convert back to probabilities
    return 1 / (1 + np.exp(-logit_sum))


def optimize_blend_weights(pred_dict: Dict[str, np.ndarray],
                           y_true: np.ndarray,
                           blend_method: str = "linear",
                           init_weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    Optimize blend weights to minimize cost metric using grid search.
    
    Args:
        pred_dict: Dict mapping model name to predictions array
        y_true: True labels
        blend_method: 'linear', 'rank', or 'logit'
        init_weights: Initial weights (default: equal)
    
    Returns:
        Optimized weights dict
    """
    names = list(pred_dict.keys())
    n_models = len(names)
    
    # Simple grid search for faster optimization
    best_cost = float("inf")
    best_weights = {n: 1.0 / n_models for n in names}
    
    # Generate weight combinations
    if n_models == 2:
        weight_options = np.linspace(0.1, 0.9, 9)
        for w1 in weight_options:
            w = [w1, 1 - w1]
            weights = {names[i]: w[i] for i in range(n_models)}
            
            if blend_method == "linear":
                blended = blend_predictions(pred_dict, weights)
            elif blend_method == "rank":
                blended = rank_blend(pred_dict, weights)
            elif blend_method == "logit":
                blended = logit_blend(pred_dict, weights)
            
            _, _, cost = find_optimal_thresholds(y_true, blended)
            if cost < best_cost:
                best_cost = cost
                best_weights = weights.copy()
    elif n_models == 3:
        for w1 in np.linspace(0.1, 0.8, 8):
            for w2 in np.linspace(0.1, 0.9 - w1, 8):
                w3 = 1 - w1 - w2
                if w3 < 0.05:
                    continue
                w = [w1, w2, w3]
                weights = {names[i]: w[i] for i in range(n_models)}
                
                if blend_method == "linear":
                    blended = blend_predictions(pred_dict, weights)
                elif blend_method == "rank":
                    blended = rank_blend(pred_dict, weights)
                elif blend_method == "logit":
                    blended = logit_blend(pred_dict, weights)
                
                _, _, cost = find_optimal_thresholds(y_true, blended)
                if cost < best_cost:
                    best_cost = cost
                    best_weights = weights.copy()
    else:
        # For more models, use equal weights
        best_weights = {n: 1.0 / n_models for n in names}
    
    return best_weights


def find_best_blend(pred_dict: Dict[str, np.ndarray],
                    y_true: np.ndarray,
                    try_methods: List[str] = None) -> Tuple[str, Dict[str, float], np.ndarray, float]:
    """
    Try multiple blending methods and return the best one.
    
    Args:
        pred_dict: Dict mapping model name to predictions array
        y_true: True labels
        try_methods: List of methods to try (default: all)
    
    Returns:
        (best_method, best_weights, best_predictions, best_cost)
    """
    if try_methods is None:
        try_methods = ["linear", "rank", "logit"]
    
    best_method = None
    best_weights = None
    best_preds = None
    best_cost = float("inf")
    
    for method in try_methods:
        print(f"\nOptimizing {method} blend...")
        
        weights = optimize_blend_weights(pred_dict, y_true, method)
        
        if method == "linear":
            preds = blend_predictions(pred_dict, weights)
        elif method == "rank":
            preds = rank_blend(pred_dict, weights)
        elif method == "logit":
            preds = logit_blend(pred_dict, weights)
        
        _, _, cost = find_optimal_thresholds(y_true, preds)
        
        print(f"{method} blend: cost = {cost:,.0f}")
        print(f"  Weights: {weights}")
        
        if cost < best_cost:
            best_cost = cost
            best_method = method
            best_weights = weights
            best_preds = preds
    
    return best_method, best_weights, best_preds, best_cost


def calibrate_predictions(preds: np.ndarray, y_true: np.ndarray,
                          method: str = "isotonic") -> Tuple[np.ndarray, object]:
    """
    Calibrate predictions using isotonic or sigmoid calibration.
    
    Args:
        preds: Predictions to calibrate
        y_true: True labels
        method: 'isotonic' or 'sigmoid'
    
    Returns:
        (calibrated_predictions, calibrator)
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(preds, y_true)
        calibrated = calibrator.predict(preds)
    elif method == "sigmoid":
        calibrator = LogisticRegression()
        calibrator.fit(preds.reshape(-1, 1), y_true)
        calibrated = calibrator.predict_proba(preds.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    return calibrated, calibrator


def apply_calibration(preds: np.ndarray, calibrator, method: str = "isotonic") -> np.ndarray:
    """Apply pre-fitted calibrator to predictions."""
    if method == "isotonic":
        return calibrator.predict(preds)
    elif method == "sigmoid":
        return calibrator.predict_proba(preds.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")
