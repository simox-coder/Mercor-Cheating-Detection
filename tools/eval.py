#!/usr/bin/env python3
"""
Local evaluation tool for Mercor Cheating Detection.

Usage:
    python tools/eval.py --oof artifacts/run_001/oof/oof_predictions.csv
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.evaluator import evaluate_oof, print_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOF predictions")
    parser.add_argument("--oof", type=str, required=True,
                        help="Path to OOF predictions CSV")
    parser.add_argument("--label-col", type=str, default="is_cheating",
                        help="Label column name")
    parser.add_argument("--pred-col", type=str, default="prediction",
                        help="Prediction column name")
    args = parser.parse_args()
    
    # Load OOF predictions
    oof_df = pd.read_csv(args.oof)
    
    print(f"Loaded {len(oof_df)} rows from {args.oof}")
    print(f"Label column: {args.label_col}")
    print(f"Prediction column: {args.pred_col}")
    print()
    
    # Evaluate
    result = evaluate_oof(oof_df, args.label_col, args.pred_col)
    print_evaluation(result)
    
    return result


if __name__ == "__main__":
    main()
