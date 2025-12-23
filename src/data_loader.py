"""Data loading and validation utilities."""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import os

from .config import (
    TRAIN_FILE, TEST_FILE, GRAPH_FILE, SAMPLE_SUB_FILE,
    FEATURE_COLS, ID_COL, LABEL_COL, HIGH_CONF_CLEAN_COL
)


def load_train(path: str = None) -> pd.DataFrame:
    """Load training data."""
    path = path or TRAIN_FILE
    df = pd.read_csv(path)
    assert ID_COL in df.columns, f"Missing {ID_COL} column"
    assert LABEL_COL in df.columns, f"Missing {LABEL_COL} column"
    assert HIGH_CONF_CLEAN_COL in df.columns, f"Missing {HIGH_CONF_CLEAN_COL} column"
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature column {col}"
    return df


def load_test(path: str = None) -> pd.DataFrame:
    """Load test data."""
    path = path or TEST_FILE
    df = pd.read_csv(path)
    assert ID_COL in df.columns, f"Missing {ID_COL} column"
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature column {col}"
    return df


def load_graph(path: str = None) -> pd.DataFrame:
    """Load social graph edges."""
    path = path or GRAPH_FILE
    df = pd.read_csv(path)
    assert "user_a" in df.columns and "user_b" in df.columns
    return df


def load_sample_submission(path: str = None) -> pd.DataFrame:
    """Load sample submission for format reference."""
    path = path or SAMPLE_SUB_FILE
    df = pd.read_csv(path)
    return df


def get_labeled_data(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract labeled subset (where is_cheating is not NaN)."""
    mask = train_df[LABEL_COL].notna()
    X = train_df.loc[mask].copy()
    y = X[LABEL_COL].values.astype(int)
    return X, y


def get_unlabeled_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Extract unlabeled subset (high_conf_clean=1, is_cheating=NaN)."""
    mask = (train_df[HIGH_CONF_CLEAN_COL] == 1.0) & (train_df[LABEL_COL].isna())
    return train_df.loc[mask].copy()


def validate_submission(sub_df: pd.DataFrame, sample_df: pd.DataFrame) -> bool:
    """Validate submission format."""
    # Check columns
    assert set(sub_df.columns) == {"user_hash", "prediction"}, \
        f"Expected columns ['user_hash', 'prediction'], got {list(sub_df.columns)}"
    
    # Check row count
    assert len(sub_df) == len(sample_df), \
        f"Row count mismatch: {len(sub_df)} vs {len(sample_df)}"
    
    # Check user_hash matches
    assert set(sub_df["user_hash"]) == set(sample_df["user_hash"]), \
        "user_hash mismatch"
    
    # Check predictions are valid
    assert sub_df["prediction"].notna().all(), "NaN predictions found"
    assert (sub_df["prediction"] >= 0).all() and (sub_df["prediction"] <= 1).all(), \
        "Predictions must be in [0, 1]"
    
    return True


def create_missing_indicators(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Create missing value indicator columns."""
    cols = cols or FEATURE_COLS
    for col in cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def fill_missing_values(df: pd.DataFrame, cols: list = None, fill_value: float = -999) -> pd.DataFrame:
    """Fill missing values with a constant."""
    cols = cols or FEATURE_COLS
    for col in cols:
        df[col] = df[col].fillna(fill_value)
    return df


def get_data_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Get basic data statistics."""
    labeled = train_df[LABEL_COL].notna()
    return {
        "train_total": len(train_df),
        "train_labeled": labeled.sum(),
        "train_unlabeled": (~labeled).sum(),
        "train_cheating": (train_df[LABEL_COL] == 1).sum(),
        "train_clean": (train_df[LABEL_COL] == 0).sum(),
        "train_high_conf_clean": (train_df[HIGH_CONF_CLEAN_COL] == 1).sum(),
        "test_total": len(test_df),
    }
