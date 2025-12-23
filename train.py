#!/usr/bin/env python3
"""
Main training script for Mercor Cheating Detection competition.

Usage:
    python train.py --data <DATA_DIR> --out artifacts/run_001

This script:
1. Loads and validates data
2. Computes graph features (degree, PageRank, components, node2vec)
3. Creates missing indicators
4. Splits data using component-based CV (leakage-safe)
5. Trains CatBoost, LightGBM, XGBoost ensemble
6. Handles high_conf_clean with PU learning
7. Optimizes ensemble weights
8. Applies score propagation
9. Saves models and produces OOF predictions
"""
import argparse
import os
import sys
import json
import pickle
import time
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FEATURE_COLS, ID_COL, LABEL_COL, HIGH_CONF_CLEAN_COL,
    SEED, N_FOLDS, PU_WEIGHT, ARTIFACTS_DIR
)
from src.data_loader import (
    load_train, load_test, load_graph, load_sample_submission,
    get_labeled_data, get_unlabeled_data, create_missing_indicators,
    fill_missing_values, validate_submission
)
from src.graph_features import (
    build_all_graph_features, compute_connected_components,
    compute_degree_features
)
from src.evaluator import evaluate_predictions, print_evaluation
from src.cv_utils import component_group_split, check_cv_leakage
from src.models import (
    train_catboost, train_lightgbm, train_xgboost,
    create_sample_weights, EnsembleModel, predict_with_model
)
from src.propagation import propagate_and_blend, build_adjacency_list
from src.blending import (
    blend_predictions, rank_blend, logit_blend,
    optimize_blend_weights, find_best_blend, calibrate_predictions
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train cheating detection models")
    parser.add_argument("--data", type=str, default=".",
                        help="Data directory containing train.csv, test.csv, etc.")
    parser.add_argument("--out", type=str, default="artifacts/run_001",
                        help="Output directory for models and results")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS,
                        help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed")
    parser.add_argument("--pu-weight", type=float, default=PU_WEIGHT,
                        help="Weight for PU learning pseudo-negatives")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip node2vec embeddings (faster)")
    parser.add_argument("--skip-propagation", action="store_true",
                        help="Skip score propagation")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: fewer iterations, skip embeddings")
    return parser.parse_args()


def setup_directories(out_dir: str):
    """Create output directories."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "oof"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)


def load_data(data_dir: str):
    """Load all data files."""
    print("Loading data...")
    
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    graph_path = os.path.join(data_dir, "social_graph.csv")
    sample_sub_path = os.path.join(data_dir, "sample_submission.csv")
    
    train_df = load_train(train_path)
    test_df = load_test(test_path)
    graph_df = load_graph(graph_path)
    sample_sub = load_sample_submission(sample_sub_path)
    
    print(f"Train: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")
    print(f"Graph: {len(graph_df)} edges")
    
    return train_df, test_df, graph_df, sample_sub


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     graph_df: pd.DataFrame, cache_dir: str,
                     compute_embeddings: bool = True):
    """Prepare all features including graph features."""
    print("\n=== Preparing features ===")
    
    # Combine train and test for feature engineering
    all_users = pd.concat([
        train_df[[ID_COL]],
        test_df[[ID_COL]]
    ]).drop_duplicates()
    
    # Compute graph features
    print("\nComputing graph features...")
    graph_feats, component_map = build_all_graph_features(
        graph_df, 
        cache_dir=cache_dir,
        compute_embeddings=compute_embeddings
    )
    
    # Merge graph features to train/test
    train_df = train_df.merge(graph_feats, on=ID_COL, how="left")
    test_df = test_df.merge(graph_feats, on=ID_COL, how="left")
    
    # Fill missing graph features (users not in graph)
    graph_cols = [c for c in graph_feats.columns if c != ID_COL]
    for col in graph_cols:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)
    
    # Create missing indicators for tabular features
    print("\nCreating missing indicators...")
    train_df = create_missing_indicators(train_df, FEATURE_COLS)
    test_df = create_missing_indicators(test_df, FEATURE_COLS)
    
    # Fill missing values
    train_df = fill_missing_values(train_df, FEATURE_COLS, fill_value=-999)
    test_df = fill_missing_values(test_df, FEATURE_COLS, fill_value=-999)
    
    # Get feature columns
    missing_cols = [f"{c}_missing" for c in FEATURE_COLS]
    all_feature_cols = FEATURE_COLS + missing_cols + graph_cols
    
    print(f"\nTotal features: {len(all_feature_cols)}")
    
    return train_df, test_df, all_feature_cols, component_map


def train_fold(X_train, y_train, X_val, y_val, 
               sample_weight=None, model_type="catboost", fast=False):
    """Train a single model fold."""
    from src.config import CATBOOST_PARAMS, LIGHTGBM_PARAMS, XGBOOST_PARAMS
    
    if model_type == "catboost":
        params = CATBOOST_PARAMS.copy()
        if fast:
            params["iterations"] = 500
        model, val_preds = train_catboost(X_train, y_train, X_val, y_val, params, sample_weight)
    elif model_type == "lightgbm":
        params = LIGHTGBM_PARAMS.copy()
        if fast:
            params["n_estimators"] = 500
        model, val_preds = train_lightgbm(X_train, y_train, X_val, y_val, params, sample_weight)
    elif model_type == "xgboost":
        params = XGBOOST_PARAMS.copy()
        if fast:
            params["n_estimators"] = 500
        model, val_preds = train_xgboost(X_train, y_train, X_val, y_val, params, sample_weight)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, val_preds


def run_cv(df: pd.DataFrame, feature_cols: list, splits: list,
           model_type: str = "catboost", pu_weight: float = PU_WEIGHT,
           fast: bool = False):
    """Run cross-validation for a model type."""
    print(f"\n=== Training {model_type} ===")
    
    X = df[feature_cols].values
    y = df[LABEL_COL].values.astype(float)
    high_conf_clean = df[HIGH_CONF_CLEAN_COL].values
    
    oof_preds = np.full(len(df), np.nan)
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold + 1}/{len(splits)} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create sample weights with PU learning
        train_hcc = high_conf_clean[train_idx]
        sample_weight = create_sample_weights(
            y_train, 
            high_conf_clean_mask=(train_hcc == 1),
            pu_weight=pu_weight
        )
        
        # Train
        model, val_preds = train_fold(
            X_train, y_train, X_val, y_val,
            sample_weight=sample_weight,
            model_type=model_type,
            fast=fast
        )
        
        oof_preds[val_idx] = val_preds
        models.append(model)
        
        # Evaluate fold
        fold_mask = ~np.isnan(y_val)
        if fold_mask.sum() > 0:
            result = evaluate_predictions(y_val[fold_mask], val_preds[fold_mask])
            print(f"Fold {fold + 1} cost: {result['total_cost']:,.0f}")
    
    return oof_preds, models


def main():
    args = parse_args()
    
    # Setup
    np.random.seed(args.seed)
    start_time = time.time()
    
    # Directories
    data_dir = os.path.abspath(args.data)
    out_dir = os.path.abspath(args.out)
    setup_directories(out_dir)
    
    # Save config
    config = vars(args)
    config["start_time"] = datetime.now().isoformat()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Load data
    train_df, test_df, graph_df, sample_sub = load_data(data_dir)
    
    # Prepare features
    compute_embeddings = not (args.skip_embeddings or args.fast)
    train_df, test_df, feature_cols, component_map = prepare_features(
        train_df, test_df, graph_df,
        cache_dir=out_dir,
        compute_embeddings=compute_embeddings
    )
    
    # Get labeled subset for training
    labeled_mask = train_df[LABEL_COL].notna()
    labeled_df = train_df[labeled_mask].copy()
    y_true = labeled_df[LABEL_COL].values.astype(int)
    
    print(f"\nLabeled samples: {len(labeled_df)}")
    print(f"Cheating rate: {y_true.mean():.2%}")
    
    # Create CV splits (component-based for leakage safety)
    print("\n=== Creating CV splits ===")
    splits = component_group_split(labeled_df, component_map, n_splits=args.n_folds)
    
    # Check for leakage
    has_leakage = check_cv_leakage(splits, labeled_df, component_map)
    if has_leakage:
        print("WARNING: CV splits have potential leakage!")
    
    # Train models
    oof_results = {}
    all_models = {}
    
    for model_type in ["catboost", "lightgbm", "xgboost"]:
        oof_preds, models = run_cv(
            labeled_df, feature_cols, splits,
            model_type=model_type,
            pu_weight=args.pu_weight,
            fast=args.fast
        )
        oof_results[model_type] = oof_preds
        all_models[model_type] = models
        
        # Evaluate
        result = evaluate_predictions(y_true, oof_preds)
        print(f"\n{model_type} OOF Results:")
        print_evaluation(result)
    
    # Find best blend
    print("\n=== Optimizing ensemble blend ===")
    best_method, best_weights, best_oof, best_cost = find_best_blend(
        oof_results, y_true
    )
    
    print(f"\nBest blend: {best_method}")
    print(f"Best weights: {best_weights}")
    print(f"Best OOF cost: {best_cost:,.0f}")
    
    # Apply propagation if enabled
    if not args.skip_propagation:
        print("\n=== Applying score propagation ===")
        
        adj_list = build_adjacency_list(graph_df)
        
        # Try different propagation methods
        best_prop_cost = best_cost
        best_prop_preds = best_oof
        best_prop_config = None
        
        for method in ["simple", "pagerank"]:
            for alpha in [0.1, 0.2, 0.3]:
                prop_preds = propagate_and_blend(
                    best_oof,
                    labeled_df[ID_COL].values,
                    graph_df,
                    prop_weight=alpha,
                    method=method
                )
                
                result = evaluate_predictions(y_true, prop_preds)
                cost = result["total_cost"]
                
                print(f"{method} alpha={alpha}: cost={cost:,.0f}")
                
                if cost < best_prop_cost:
                    best_prop_cost = cost
                    best_prop_preds = prop_preds
                    best_prop_config = {"method": method, "alpha": alpha}
        
        if best_prop_config:
            print(f"\nBest propagation: {best_prop_config}")
            print(f"Propagated cost: {best_prop_cost:,.0f}")
            best_oof = best_prop_preds
    
    # Final OOF evaluation
    print("\n=== Final OOF Evaluation ===")
    final_result = evaluate_predictions(y_true, best_oof)
    print_evaluation(final_result)
    
    # Save OOF predictions
    oof_df = labeled_df[[ID_COL, LABEL_COL]].copy()
    oof_df["prediction"] = best_oof
    for model_type, preds in oof_results.items():
        oof_df[f"{model_type}_pred"] = preds
    oof_df.to_csv(os.path.join(out_dir, "oof", "oof_predictions.csv"), index=False)
    
    # Save models
    print("\n=== Saving models ===")
    for model_type, models in all_models.items():
        for i, model in enumerate(models):
            model_path = os.path.join(out_dir, "models", f"{model_type}_fold{i}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
    
    # Save blend config
    blend_config = {
        "method": best_method,
        "weights": best_weights,
        "propagation": best_prop_config if not args.skip_propagation else None,
        "cv_cost": final_result["total_cost"],
        "cv_score": final_result["cost_score"],
        "low_thresh": final_result["low_thresh"],
        "high_thresh": final_result["high_thresh"],
    }
    with open(os.path.join(out_dir, "blend_config.json"), "w") as f:
        json.dump(blend_config, f, indent=2)
    
    # Save feature list
    with open(os.path.join(out_dir, "features.json"), "w") as f:
        json.dump({"features": feature_cols}, f, indent=2)
    
    # Save component map
    with open(os.path.join(out_dir, "component_map.pkl"), "wb") as f:
        pickle.dump(component_map, f)
    
    # Generate test predictions
    print("\n=== Generating test predictions ===")
    X_test = test_df[feature_cols].values
    
    test_preds = {}
    for model_type, models in all_models.items():
        preds = np.zeros(len(test_df))
        for model in models:
            preds += predict_with_model(model, X_test) / len(models)
        test_preds[model_type] = preds
    
    # Blend test predictions
    if best_method == "linear":
        final_test_preds = blend_predictions(test_preds, best_weights)
    elif best_method == "rank":
        final_test_preds = rank_blend(test_preds, best_weights)
    elif best_method == "logit":
        final_test_preds = logit_blend(test_preds, best_weights)
    
    # Apply propagation to test
    if not args.skip_propagation and best_prop_config:
        final_test_preds = propagate_and_blend(
            final_test_preds,
            test_df[ID_COL].values,
            graph_df,
            prop_weight=best_prop_config["alpha"],
            method=best_prop_config["method"]
        )
    
    # Create submission
    submission = pd.DataFrame({
        "user_hash": test_df[ID_COL],
        "prediction": final_test_preds
    })
    
    # Validate submission
    validate_submission(submission, sample_sub)
    
    # Save submission
    sub_path = os.path.join(out_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    
    # Compute checksum
    with open(sub_path, "rb") as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Time: {elapsed:.1f}s")
    print(f"CV Score: {final_result['cost_score']:,.0f}")
    print(f"Submission: {sub_path}")
    print(f"Checksum: {checksum}")
    print("=" * 50)
    
    # Update scoreboard
    scoreboard_path = os.path.join(ARTIFACTS_DIR, "scoreboard.csv")
    scoreboard_entry = {
        "run_id": os.path.basename(out_dir),
        "timestamp": datetime.now().isoformat(),
        "cv_cost": final_result["total_cost"],
        "cv_score": final_result["cost_score"],
        "public_lb_score": None,
        "target_lb_score": None,
        "beat_top1": False,
        "notes": f"{best_method} blend, prop={best_prop_config}"
    }
    
    if os.path.exists(scoreboard_path):
        scoreboard = pd.read_csv(scoreboard_path)
        scoreboard = pd.concat([scoreboard, pd.DataFrame([scoreboard_entry])], ignore_index=True)
    else:
        scoreboard = pd.DataFrame([scoreboard_entry])
    
    scoreboard.to_csv(scoreboard_path, index=False)
    
    return final_result


if __name__ == "__main__":
    main()
