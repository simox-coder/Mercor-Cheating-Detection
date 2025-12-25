"""
Data Loader Module
==================
Common data loading and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import json

# Global seed for reproducibility
SEED = 42


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_train_data(data_dir: str = '.') -> pd.DataFrame:
    """Load training data."""
    return pd.read_csv(Path(data_dir) / 'train.csv')


def load_test_data(data_dir: str = '.') -> pd.DataFrame:
    """Load test data."""
    return pd.read_csv(Path(data_dir) / 'test.csv')


def load_graph(data_dir: str = '.') -> pd.DataFrame:
    """Load social graph edge list."""
    return pd.read_csv(Path(data_dir) / 'social_graph.csv')


def load_metadata(data_dir: str = '.') -> Dict:
    """Load feature metadata."""
    with open(Path(data_dir) / 'feature_metadata.json', 'r') as f:
        return json.load(f)


def get_feature_columns() -> List[str]:
    """Return list of feature column names."""
    return [f'feature_{i:03d}' for i in range(1, 19)]


def get_labeled_mask(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for labeled rows."""
    return df['is_cheating'].notna()


def get_unlabeled_mask(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for unlabeled rows (high_conf_clean=1)."""
    return df['is_cheating'].isna() & (df['high_conf_clean'] == 1)


def split_labeled_unlabeled(train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split train into labeled and unlabeled DataFrames.
    
    Returns:
        labeled: DataFrame with is_cheating not NaN
        unlabeled: DataFrame with is_cheating NaN and high_conf_clean=1
    """
    labeled_mask = get_labeled_mask(train)
    unlabeled_mask = get_unlabeled_mask(train)
    
    return train[labeled_mask].copy(), train[unlabeled_mask].copy()


def create_proxy_split(
    labeled: pd.DataFrame,
    seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified 50/50 public/private proxy split.
    
    Args:
        labeled: Labeled training data
        seed: Random seed
    
    Returns:
        public_proxy: 50% of labeled data
        private_proxy: 50% of labeled data
    """
    from sklearn.model_selection import train_test_split
    
    public, private = train_test_split(
        labeled,
        test_size=0.5,
        stratify=labeled['is_cheating'],
        random_state=seed
    )
    
    return public.reset_index(drop=True), private.reset_index(drop=True)


def create_features(
    df: pd.DataFrame,
    add_missing_flags: bool = True,
    add_row_stats: bool = True
) -> pd.DataFrame:
    """
    Create feature matrix with optional missing indicators and row statistics.
    
    Args:
        df: Input DataFrame
        add_missing_flags: Add per-feature missingness indicators
        add_row_stats: Add row-level statistics
    
    Returns:
        Feature DataFrame
    """
    feature_cols = get_feature_columns()
    
    # Start with base features
    features = df[feature_cols].copy()
    
    if add_missing_flags:
        for col in feature_cols:
            features[f'{col}_missing'] = df[col].isna().astype(int)
    
    if add_row_stats:
        # Row-level statistics (computed on original feature values)
        numeric_features = df[feature_cols]
        features['na_count'] = numeric_features.isna().sum(axis=1)
        features['row_mean'] = numeric_features.mean(axis=1)
        features['row_std'] = numeric_features.std(axis=1)
        features['row_min'] = numeric_features.min(axis=1)
        features['row_max'] = numeric_features.max(axis=1)
        features['row_range'] = features['row_max'] - features['row_min']
    
    return features


def print_data_summary(train: pd.DataFrame, test: pd.DataFrame, graph: pd.DataFrame):
    """Print summary of loaded data."""
    labeled, unlabeled = split_labeled_unlabeled(train)
    
    print("="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Train rows:      {len(train):>10,}")
    print(f"  Labeled:       {len(labeled):>10,}")
    print(f"  Unlabeled:     {len(unlabeled):>10,}")
    print(f"Test rows:       {len(test):>10,}")
    print(f"Graph edges:     {len(graph):>10,}")
    print()
    print(f"Cheater rate (labeled): {labeled['is_cheating'].mean():.4f}")
    print(f"Cheaters:        {int(labeled['is_cheating'].sum()):>10,}")
    print(f"Non-cheaters:    {int((labeled['is_cheating'] == 0).sum()):>10,}")
    print("="*60)


if __name__ == '__main__':
    # Test data loading
    train = load_train_data()
    test = load_test_data()
    graph = load_graph()
    metadata = load_metadata()
    
    print_data_summary(train, test, graph)
    
    # Test feature creation
    features = create_features(train)
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)[:10]}...")
