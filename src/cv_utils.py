"""Cross-validation utilities with leakage-safe component-based splitting."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Generator
from sklearn.model_selection import StratifiedKFold, GroupKFold

from .config import N_FOLDS, SEED


def component_group_split(df: pd.DataFrame, component_map: Dict[str, int],
                          n_splits: int = N_FOLDS, 
                          random_state: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create CV splits based on connected components.
    Ensures all users in the same component are in the same fold.
    This prevents graph leakage.
    """
    # Map user_hash to component_id
    df = df.copy()
    df["component_id"] = df["user_hash"].map(component_map)
    
    # For users not in graph, assign unique component IDs
    max_comp = max(component_map.values()) + 1 if component_map else 0
    missing_mask = df["component_id"].isna()
    df.loc[missing_mask, "component_id"] = np.arange(max_comp, max_comp + missing_mask.sum())
    df["component_id"] = df["component_id"].astype(int)
    
    # Group-based split by component
    groups = df["component_id"].values
    gkf = GroupKFold(n_splits=n_splits)
    
    splits = []
    for train_idx, val_idx in gkf.split(df, groups=groups):
        splits.append((train_idx, val_idx))
    
    return splits


def stratified_split(df: pd.DataFrame, label_col: str = "is_cheating",
                     n_splits: int = N_FOLDS,
                     random_state: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Standard stratified K-fold split (no leakage protection)."""
    y = df[label_col].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, val_idx in skf.split(df, y):
        splits.append((train_idx, val_idx))
    
    return splits


def create_oof_predictions(df: pd.DataFrame, 
                           splits: List[Tuple[np.ndarray, np.ndarray]],
                           train_func,
                           feature_cols: list,
                           label_col: str = "is_cheating") -> Tuple[np.ndarray, list]:
    """
    Create out-of-fold predictions.
    
    Args:
        df: DataFrame with features and labels
        splits: List of (train_idx, val_idx) tuples
        train_func: Function(X_train, y_train, X_val, y_val) -> (model, val_preds)
        feature_cols: List of feature columns to use
        label_col: Label column name
    
    Returns:
        oof_preds: Array of OOF predictions
        models: List of trained models
    """
    oof_preds = np.zeros(len(df))
    models = []
    
    X = df[feature_cols].values
    y = df[label_col].values
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n=== Fold {fold + 1}/{len(splits)} ===")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model, val_preds = train_func(X_train, y_train, X_val, y_val)
        
        oof_preds[val_idx] = val_preds
        models.append(model)
    
    return oof_preds, models


def check_cv_leakage(splits: List[Tuple[np.ndarray, np.ndarray]], 
                     df: pd.DataFrame,
                     component_map: Dict[str, int]) -> bool:
    """
    Check if CV splits have graph leakage.
    Returns True if leakage detected.
    """
    df = df.copy()
    df["component_id"] = df["user_hash"].map(component_map).fillna(-1).astype(int)
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_comps = set(df.iloc[train_idx]["component_id"].unique())
        val_comps = set(df.iloc[val_idx]["component_id"].unique())
        
        # Check for component overlap
        overlap = train_comps & val_comps
        overlap = overlap - {-1}  # Exclude isolated nodes
        
        if overlap:
            print(f"WARNING: Fold {fold} has {len(overlap)} overlapping components")
            return True
    
    print("CV splits are leakage-safe (no component overlap)")
    return False
