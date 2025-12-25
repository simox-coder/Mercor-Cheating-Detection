"""
Branch A: Tabular GBDT Cost-Optimized
=====================================
Strongest non-graph baseline using LightGBM + XGBoost + CatBoost ensemble.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.preprocessing import StandardScaler
import optuna
from tqdm import tqdm

from data_loader import (
    SEED, load_train_data, load_test_data, load_metadata,
    get_feature_columns, create_features, split_labeled_unlabeled,
    create_proxy_split, set_seed
)
from cv_harness import (
    cv_train_evaluate, train_final_model, evaluate_on_subset,
    save_cv_artifacts, create_cv_folds
)
import metric

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_tabular_features(df: pd.DataFrame) -> np.ndarray:
    """Create full tabular feature matrix for Branch A."""
    features = create_features(df, add_missing_flags=True, add_row_stats=True)
    return features.values


def get_lgbm_model(params: Optional[Dict] = None, seed: int = SEED) -> LGBMClassifier:
    """Create LightGBM model with given or default parameters."""
    default_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': seed,
        'verbosity': -1,
        'n_jobs': -1
    }
    if params:
        default_params.update(params)
    return LGBMClassifier(**default_params)


def get_xgb_model(params: Optional[Dict] = None, seed: int = SEED) -> XGBClassifier:
    """Create XGBoost model with given or default parameters."""
    default_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': seed,
        'verbosity': 0,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    if params:
        default_params.update(params)
    return XGBClassifier(**default_params)


def get_cat_model(params: Optional[Dict] = None, seed: int = SEED) -> CatBoostClassifier:
    """Create CatBoost model with given or default parameters."""
    default_params = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': seed,
        'verbose': False,
        'thread_count': -1
    }
    if params:
        default_params.update(params)
    return CatBoostClassifier(**default_params)


def blend_predictions(
    preds_list: list,
    weights: Optional[list] = None,
    method: str = 'average'
) -> np.ndarray:
    """
    Blend multiple predictions.
    
    Args:
        preds_list: List of prediction arrays
        weights: Optional weights for weighted average
        method: 'average', 'weighted', or 'rank'
    
    Returns:
        Blended predictions
    """
    preds_list = [np.array(p) for p in preds_list]
    
    if method == 'average':
        return np.mean(preds_list, axis=0)
    elif method == 'weighted':
        if weights is None:
            weights = [1.0 / len(preds_list)] * len(preds_list)
        return np.average(preds_list, axis=0, weights=weights)
    elif method == 'rank':
        # Convert to ranks, average, then rescale
        from scipy.stats import rankdata
        ranks = [rankdata(p) / len(p) for p in preds_list]
        return np.mean(ranks, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_branch_a(
    output_dir: str = 'artifacts/branch_A_tabular',
    verbose: bool = True,
    quick_mode: bool = False  # For testing
) -> Dict:
    """
    Run Branch A: Tabular GBDT pipeline.
    
    Args:
        output_dir: Directory for artifacts
        verbose: Print progress
        quick_mode: Use fewer iterations for testing
    
    Returns:
        Dictionary with results summary
    """
    set_seed(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("="*60)
        print("BRANCH A: TABULAR GBDT COST-OPTIMIZED")
        print("="*60)
    
    # Load data
    train = load_train_data()
    test = load_test_data()
    
    # Split labeled/unlabeled
    labeled, unlabeled = split_labeled_unlabeled(train)
    
    if verbose:
        print(f"Labeled samples: {len(labeled):,}")
        print(f"Unlabeled samples: {len(unlabeled):,}")
        print(f"Test samples: {len(test):,}")
    
    # Create features
    X_labeled = create_tabular_features(labeled)
    X_unlabeled = create_tabular_features(unlabeled)
    X_test = create_tabular_features(test)
    y_labeled = labeled['is_cheating'].values
    
    # Handle NaN values
    X_labeled = np.nan_to_num(X_labeled, nan=-999)
    X_unlabeled = np.nan_to_num(X_unlabeled, nan=-999)
    X_test = np.nan_to_num(X_test, nan=-999)
    
    # Create proxy split
    public_proxy, private_proxy = create_proxy_split(labeled, seed=SEED)
    public_idx = labeled['user_hash'].isin(public_proxy['user_hash']).values
    private_idx = labeled['user_hash'].isin(private_proxy['user_hash']).values
    
    if verbose:
        print(f"Public proxy: {public_idx.sum():,} samples")
        print(f"Private proxy: {private_idx.sum():,} samples")
    
    # Model parameters (can be tuned with Optuna)
    n_estimators = 100 if quick_mode else 500
    
    # Train individual models with CV
    models_results = {}
    oof_predictions = {}
    
    # LightGBM
    if verbose:
        print("\n--- LightGBM ---")
    lgbm_oof, lgbm_summary = cv_train_evaluate(
        X_labeled, y_labeled, X_unlabeled,
        lambda: get_lgbm_model({'n_estimators': n_estimators}),
        n_folds=5,
        pseudo_weight=0.01,
        verbose=verbose
    )
    oof_predictions['lgbm'] = lgbm_oof
    models_results['lgbm'] = lgbm_summary
    
    # XGBoost
    if verbose:
        print("\n--- XGBoost ---")
    xgb_oof, xgb_summary = cv_train_evaluate(
        X_labeled, y_labeled, X_unlabeled,
        lambda: get_xgb_model({'n_estimators': n_estimators}),
        n_folds=5,
        pseudo_weight=0.01,
        verbose=verbose
    )
    oof_predictions['xgb'] = xgb_oof
    models_results['xgb'] = xgb_summary
    
    # CatBoost
    if verbose:
        print("\n--- CatBoost ---")
    cat_oof, cat_summary = cv_train_evaluate(
        X_labeled, y_labeled, X_unlabeled,
        lambda: get_cat_model({'iterations': n_estimators}),
        n_folds=5,
        pseudo_weight=0.01,
        verbose=verbose
    )
    oof_predictions['cat'] = cat_oof
    models_results['cat'] = cat_summary
    
    # Try different blending methods
    blend_results = {}
    
    if verbose:
        print("\n--- Blending ---")
    
    # Simple average
    avg_blend = blend_predictions([lgbm_oof, xgb_oof, cat_oof], method='average')
    avg_result = metric.score(y_labeled, avg_blend)
    blend_results['average'] = {
        'oof_cost': avg_result['best_cost'],
        't_low': avg_result['t_low'],
        't_high': avg_result['t_high']
    }
    if verbose:
        print(f"  Average blend: cost={avg_result['best_cost']:,}")
    
    # Weighted average (based on individual costs - lower is better)
    costs = [lgbm_summary['oof_cost'], xgb_summary['oof_cost'], cat_summary['oof_cost']]
    inv_costs = [1.0 / (c + 1) for c in costs]
    weights = [w / sum(inv_costs) for w in inv_costs]
    weighted_blend = blend_predictions([lgbm_oof, xgb_oof, cat_oof], weights=weights, method='weighted')
    weighted_result = metric.score(y_labeled, weighted_blend)
    blend_results['weighted'] = {
        'oof_cost': weighted_result['best_cost'],
        'weights': weights,
        't_low': weighted_result['t_low'],
        't_high': weighted_result['t_high']
    }
    if verbose:
        print(f"  Weighted blend: cost={weighted_result['best_cost']:,} (weights={[f'{w:.3f}' for w in weights]})")
    
    # Rank averaging
    rank_blend = blend_predictions([lgbm_oof, xgb_oof, cat_oof], method='rank')
    rank_result = metric.score(y_labeled, rank_blend)
    blend_results['rank'] = {
        'oof_cost': rank_result['best_cost'],
        't_low': rank_result['t_low'],
        't_high': rank_result['t_high']
    }
    if verbose:
        print(f"  Rank blend: cost={rank_result['best_cost']:,}")
    
    # Choose best blend method
    best_blend_method = min(blend_results.keys(), key=lambda k: blend_results[k]['oof_cost'])
    best_oof_cost = blend_results[best_blend_method]['oof_cost']
    
    if verbose:
        print(f"\nBest blend method: {best_blend_method} (cost={best_oof_cost:,})")
    
    # Evaluate on proxy splits
    if best_blend_method == 'average':
        final_oof = avg_blend
    elif best_blend_method == 'weighted':
        final_oof = weighted_blend
    else:
        final_oof = rank_blend
    
    public_result = evaluate_on_subset(y_labeled, final_oof, public_idx, "public_proxy")
    private_result = evaluate_on_subset(y_labeled, final_oof, private_idx, "private_proxy")
    
    if verbose:
        print(f"\nProxy evaluation:")
        print(f"  Public proxy cost:  {public_result['best_cost']:,}")
        print(f"  Private proxy cost: {private_result['best_cost']:,}")
    
    # Train final models on all data
    if verbose:
        print("\n--- Training final models ---")
    
    final_lgbm = train_final_model(
        X_labeled, y_labeled, X_unlabeled,
        lambda: get_lgbm_model({'n_estimators': n_estimators}),
        pseudo_weight=0.01
    )
    
    final_xgb = train_final_model(
        X_labeled, y_labeled, X_unlabeled,
        lambda: get_xgb_model({'n_estimators': n_estimators}),
        pseudo_weight=0.01
    )
    
    final_cat = train_final_model(
        X_labeled, y_labeled, X_unlabeled,
        lambda: get_cat_model({'iterations': n_estimators}),
        pseudo_weight=0.01
    )
    
    # Get test predictions
    test_lgbm = final_lgbm.predict_proba(X_test)[:, 1]
    test_xgb = final_xgb.predict_proba(X_test)[:, 1]
    test_cat = final_cat.predict_proba(X_test)[:, 1]
    
    # Blend test predictions
    if best_blend_method == 'average':
        test_preds = blend_predictions([test_lgbm, test_xgb, test_cat], method='average')
    elif best_blend_method == 'weighted':
        test_preds = blend_predictions([test_lgbm, test_xgb, test_cat], weights=weights, method='weighted')
    else:
        test_preds = blend_predictions([test_lgbm, test_xgb, test_cat], method='rank')
    
    # Save artifacts
    save_cv_artifacts(
        final_oof,
        labeled['user_hash'].values,
        blend_results[best_blend_method],
        output_dir
    )
    
    # Save proxy results
    with open(output_dir / 'proxy_public.json', 'w') as f:
        json.dump({
            'cost': public_result['best_cost'],
            'score': public_result['best_score'],
            't_low': public_result['t_low'],
            't_high': public_result['t_high'],
            'n_samples': int(public_idx.sum())
        }, f, indent=2)
    
    with open(output_dir / 'proxy_private.json', 'w') as f:
        json.dump({
            'cost': private_result['best_cost'],
            'score': private_result['best_score'],
            't_low': private_result['t_low'],
            't_high': private_result['t_high'],
            'n_samples': int(private_idx.sum())
        }, f, indent=2)
    
    # Create submission
    submission = pd.DataFrame({
        'user_hash': test['user_hash'],
        'prediction': test_preds
    })
    submission.to_csv('submission_A.csv', index=False)
    
    if verbose:
        print(f"\nSubmission saved to: submission_A.csv")
        print(f"Prediction range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    
    # Determine optimal thresholds for submission
    oof_eval = metric.score(y_labeled, final_oof)
    
    results = {
        'branch': 'A',
        'method': 'Tabular GBDT Ensemble',
        'oof_cost': best_oof_cost,
        'oof_score': -best_oof_cost,
        'public_proxy_cost': public_result['best_cost'],
        'private_proxy_cost': private_result['best_cost'],
        't_low': oof_eval['t_low'],
        't_high': oof_eval['t_high'],
        'blend_method': best_blend_method,
        'models': list(models_results.keys()),
        'submission_file': 'submission_A.csv'
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print("\n" + "="*60)
        print("BRANCH A COMPLETE")
        print(f"  OOF Cost: {best_oof_cost:,}")
        print(f"  Public Proxy Cost: {public_result['best_cost']:,}")
        print(f"  Private Proxy Cost: {private_result['best_cost']:,}")
        print("="*60)
    
    return results


if __name__ == '__main__':
    results = run_branch_a(verbose=True, quick_mode=False)
    print(json.dumps(results, indent=2))
