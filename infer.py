#!/usr/bin/env python3
"""
Inference script for Mercor Cheating Detection competition.

Usage:
    python infer.py --data <DATA_DIR> --ckpt artifacts/run_001 --out submission.csv
"""
import argparse
import os
import sys
import json
import pickle
import hashlib

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FEATURE_COLS, ID_COL, SEED
from src.data_loader import (
    load_test, load_graph, load_sample_submission,
    create_missing_indicators, fill_missing_values, validate_submission
)
from src.graph_features import build_all_graph_features
from src.models import predict_with_model
from src.propagation import propagate_and_blend
from src.blending import blend_predictions, rank_blend, logit_blend


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions")
    parser.add_argument("--data", type=str, default=".",
                        help="Data directory containing test.csv, social_graph.csv")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint directory with trained models")
    parser.add_argument("--out", type=str, default="submission.csv",
                        help="Output submission file path")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(SEED)
    
    data_dir = os.path.abspath(args.data)
    ckpt_dir = os.path.abspath(args.ckpt)
    
    # Load test data
    print("Loading data...")
    test_df = load_test(os.path.join(data_dir, "test.csv"))
    graph_df = load_graph(os.path.join(data_dir, "social_graph.csv"))
    sample_sub = load_sample_submission(os.path.join(data_dir, "sample_submission.csv"))
    
    print(f"Test: {len(test_df)} rows")
    
    # Load config
    with open(os.path.join(ckpt_dir, "blend_config.json")) as f:
        blend_config = json.load(f)
    
    with open(os.path.join(ckpt_dir, "features.json")) as f:
        feature_cols = json.load(f)["features"]
    
    # Compute graph features
    print("Computing graph features...")
    graph_feats, _ = build_all_graph_features(
        graph_df,
        cache_dir=ckpt_dir,
        compute_embeddings=os.path.exists(os.path.join(ckpt_dir, "node2vec_embeddings.pkl"))
    )
    
    # Merge graph features
    test_df = test_df.merge(graph_feats, on=ID_COL, how="left")
    
    # Fill missing graph features
    graph_cols = [c for c in graph_feats.columns if c != ID_COL]
    for col in graph_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(0)
    
    # Create missing indicators
    test_df = create_missing_indicators(test_df, FEATURE_COLS)
    test_df = fill_missing_values(test_df, FEATURE_COLS, fill_value=-999)
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    
    X_test = test_df[feature_cols].values
    
    # Load models and predict
    print("Loading models and predicting...")
    model_types = ["catboost", "lightgbm", "xgboost"]
    test_preds = {}
    
    for model_type in model_types:
        preds = np.zeros(len(test_df))
        fold = 0
        
        while True:
            model_path = os.path.join(ckpt_dir, "models", f"{model_type}_fold{fold}.pkl")
            if not os.path.exists(model_path):
                break
            
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            preds += predict_with_model(model, X_test)
            fold += 1
        
        if fold > 0:
            preds /= fold
            test_preds[model_type] = preds
            print(f"  {model_type}: {fold} folds loaded")
    
    # Blend predictions
    print("Blending predictions...")
    best_method = blend_config["method"]
    best_weights = blend_config["weights"]
    
    if best_method == "linear":
        final_preds = blend_predictions(test_preds, best_weights)
    elif best_method == "rank":
        final_preds = rank_blend(test_preds, best_weights)
    elif best_method == "logit":
        final_preds = logit_blend(test_preds, best_weights)
    
    # Apply propagation if configured
    prop_config = blend_config.get("propagation")
    if prop_config:
        print(f"Applying propagation: {prop_config}")
        final_preds = propagate_and_blend(
            final_preds,
            test_df[ID_COL].values,
            graph_df,
            prop_weight=prop_config["alpha"],
            method=prop_config["method"]
        )
    
    # Create submission
    submission = pd.DataFrame({
        "user_hash": test_df[ID_COL],
        "prediction": final_preds
    })
    
    # Validate
    validate_submission(submission, sample_sub)
    
    # Save
    submission.to_csv(args.out, index=False)
    
    # Checksum
    with open(args.out, "rb") as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    
    print(f"\nSubmission saved: {args.out}")
    print(f"Rows: {len(submission)}")
    print(f"Prediction range: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()
