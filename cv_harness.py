"""
CV Harness Module
=================
Leakage-safe cross-validation training harness shared by all branches.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Callable, Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

from data_loader import SEED, get_feature_columns, create_features
import metric


def create_cv_folds(
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = SEED
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified K-fold splits.
    
    Args:
        y: Target labels
        n_folds: Number of folds
        seed: Random seed
    
    Returns:
        List of (train_idx, val_idx) tuples
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def cv_train_evaluate(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: Optional[np.ndarray],
    model_factory: Callable[[], Any],
    n_folds: int = 5,
    pseudo_weight: float = 0.0,
    seed: int = SEED,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Leakage-safe cross-validation with optional pseudo-negatives.
    
    Args:
        X_labeled: Features for labeled samples
        y_labeled: Labels (0 or 1)
        X_unlabeled: Features for unlabeled samples (high_conf_clean)
        model_factory: Function that returns a new model instance
        n_folds: Number of CV folds
        pseudo_weight: Weight for pseudo-negative samples (0 = don't use)
        seed: Random seed
        verbose: Print progress
    
    Returns:
        oof_preds: Out-of-fold predictions for all labeled samples
        summary: Dictionary with CV summary statistics
    """
    folds = create_cv_folds(y_labeled, n_folds, seed)
    oof_preds = np.zeros(len(y_labeled))
    fold_costs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_folds}...", end=" ")
        
        X_train = X_labeled[train_idx]
        y_train = y_labeled[train_idx]
        X_val = X_labeled[val_idx]
        y_val = y_labeled[val_idx]
        
        # Optionally add unlabeled as pseudo-negatives
        if X_unlabeled is not None and pseudo_weight > 0:
            # Sample weights: 1 for labeled, pseudo_weight for unlabeled
            sample_weight = np.ones(len(y_train))
            
            # Add unlabeled as pseudo-negatives (y=0)
            X_train = np.vstack([X_train, X_unlabeled])
            y_train = np.concatenate([y_train, np.zeros(len(X_unlabeled))])
            sample_weight = np.concatenate([
                sample_weight, 
                np.full(len(X_unlabeled), pseudo_weight)
            ])
        else:
            sample_weight = None
        
        # Train model
        model = model_factory()
        
        # Check if model supports sample_weight
        if sample_weight is not None:
            try:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            except TypeError:
                # Fallback without sample weight
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        # Predict on validation
        if hasattr(model, 'predict_proba'):
            fold_preds = model.predict_proba(X_val)[:, 1]
        else:
            fold_preds = model.predict(X_val)
        
        oof_preds[val_idx] = fold_preds
        
        # Evaluate fold
        fold_result = metric.score(y_val, fold_preds)
        fold_costs.append(fold_result['best_cost'])
        
        if verbose:
            print(f"cost={fold_result['best_cost']:,}")
    
    # Final OOF evaluation
    oof_result = metric.score(y_labeled, oof_preds)
    
    summary = {
        'fold_costs': fold_costs,
        'mean_fold_cost': np.mean(fold_costs),
        'std_fold_cost': np.std(fold_costs),
        'oof_cost': oof_result['best_cost'],
        'oof_score': oof_result['best_score'],
        't_low': oof_result['t_low'],
        't_high': oof_result['t_high'],
        'n_pass': oof_result['n_pass'],
        'n_review': oof_result['n_review'],
        'n_block': oof_result['n_block'],
        'n_fn': oof_result['n_fn'],
        'n_fp': oof_result['n_fp']
    }
    
    if verbose:
        print(f"  OOF Cost: {oof_result['best_cost']:,}")
        print(f"  Thresholds: t_low={oof_result['t_low']:.3f}, t_high={oof_result['t_high']:.3f}")
    
    return oof_preds, summary


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_unlabeled: Optional[np.ndarray],
    model_factory: Callable[[], Any],
    pseudo_weight: float = 0.0
) -> Any:
    """
    Train final model on all labeled data.
    
    Args:
        X_train: All labeled features
        y_train: All labeled targets
        X_unlabeled: Unlabeled features for pseudo-negatives
        model_factory: Function returning new model instance
        pseudo_weight: Weight for pseudo-negatives
    
    Returns:
        Trained model
    """
    if X_unlabeled is not None and pseudo_weight > 0:
        sample_weight = np.ones(len(y_train))
        X_train = np.vstack([X_train, X_unlabeled])
        y_train = np.concatenate([y_train, np.zeros(len(X_unlabeled))])
        sample_weight = np.concatenate([
            sample_weight,
            np.full(len(X_unlabeled), pseudo_weight)
        ])
    else:
        sample_weight = None
    
    model = model_factory()
    
    if sample_weight is not None:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    
    return model


def evaluate_on_subset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    name: str = "subset"
) -> Dict:
    """
    Evaluate predictions on a subset defined by mask.
    
    Args:
        y_true: All ground truth labels
        y_pred: All predictions
        mask: Boolean mask for subset
        name: Name for logging
    
    Returns:
        Evaluation result dictionary
    """
    y_true_sub = y_true[mask]
    y_pred_sub = y_pred[mask]
    
    result = metric.score(y_true_sub, y_pred_sub)
    result['name'] = name
    result['n_samples'] = int(np.sum(mask))
    
    return result


def save_cv_artifacts(
    oof_preds: np.ndarray,
    user_hashes: np.ndarray,
    summary: Dict,
    output_dir: str,
    prefix: str = ""
):
    """
    Save CV artifacts to disk.
    
    Args:
        oof_preds: Out-of-fold predictions
        user_hashes: User hash identifiers
        summary: CV summary dictionary
        output_dir: Output directory path
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'user_hash': user_hashes,
        'prediction': oof_preds
    })
    oof_df.to_csv(output_dir / f'{prefix}oof.csv', index=False)
    
    # Save summary
    with open(output_dir / f'{prefix}cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def load_cv_artifacts(
    output_dir: str,
    prefix: str = ""
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load CV artifacts from disk.
    
    Args:
        output_dir: Directory containing artifacts
        prefix: Filename prefix
    
    Returns:
        oof_df: OOF predictions DataFrame
        summary: CV summary dictionary
    """
    output_dir = Path(output_dir)
    
    oof_df = pd.read_csv(output_dir / f'{prefix}oof.csv')
    with open(output_dir / f'{prefix}cv_summary.json', 'r') as f:
        summary = json.load(f)
    
    return oof_df, summary
