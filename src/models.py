"""Model training functions for CatBoost, LightGBM, XGBoost."""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import pickle
import os

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb

from .config import (
    CATBOOST_PARAMS, LIGHTGBM_PARAMS, XGBOOST_PARAMS,
    SEED, PU_WEIGHT
)


def train_catboost(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   params: dict = None,
                   sample_weight: np.ndarray = None) -> Tuple[CatBoostClassifier, np.ndarray]:
    """Train CatBoost classifier."""
    params = params or CATBOOST_PARAMS.copy()
    
    train_pool = Pool(X_train, y_train, weight=sample_weight)
    val_pool = Pool(X_val, y_val)
    
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, verbose=params.get("verbose", 100))
    
    val_preds = model.predict_proba(X_val)[:, 1]
    return model, val_preds


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   params: dict = None,
                   sample_weight: np.ndarray = None) -> Tuple[lgb.LGBMClassifier, np.ndarray]:
    """Train LightGBM classifier."""
    params = params or LIGHTGBM_PARAMS.copy()
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weight,
        callbacks=callbacks
    )
    
    val_preds = model.predict_proba(X_val)[:, 1]
    return model, val_preds


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  params: dict = None,
                  sample_weight: np.ndarray = None) -> Tuple[xgb.XGBClassifier, np.ndarray]:
    """Train XGBoost classifier."""
    params = params or XGBOOST_PARAMS.copy()
    
    model = xgb.XGBClassifier(**params, early_stopping_rounds=100)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weight,
        verbose=100
    )
    
    val_preds = model.predict_proba(X_val)[:, 1]
    return model, val_preds


def get_trainer(model_type: str):
    """Get trainer function for a model type."""
    trainers = {
        "catboost": train_catboost,
        "lightgbm": train_lightgbm,
        "xgboost": train_xgboost,
    }
    return trainers[model_type]


def predict_with_model(model, X: np.ndarray) -> np.ndarray:
    """Get predictions from a trained model."""
    return model.predict_proba(X)[:, 1]


def predict_ensemble(models: list, X: np.ndarray, weights: list = None) -> np.ndarray:
    """Get ensemble predictions from multiple models."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    preds = np.zeros(len(X))
    for model, w in zip(models, weights):
        preds += w * predict_with_model(model, X)
    
    return preds


def save_model(model, path: str):
    """Save model to disk."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    """Load model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def create_sample_weights(y: np.ndarray, 
                          high_conf_clean_mask: np.ndarray = None,
                          pu_weight: float = PU_WEIGHT,
                          class_weight: str = "balanced") -> np.ndarray:
    """
    Create sample weights for training.
    
    Args:
        y: Labels (0, 1, or -1 for unlabeled)
        high_conf_clean_mask: Boolean mask for high_conf_clean samples
        pu_weight: Weight for pseudo-negative samples
        class_weight: 'balanced' or None
    
    Returns:
        Sample weights array
    """
    weights = np.ones(len(y), dtype=float)
    
    # Apply class weights if requested
    if class_weight == "balanced":
        # Only count labeled samples for class weights
        labeled_mask = (y == 0) | (y == 1)
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        
        if n_pos > 0 and n_neg > 0:
            pos_weight = len(y[labeled_mask]) / (2 * n_pos)
            neg_weight = len(y[labeled_mask]) / (2 * n_neg)
            
            weights[y == 1] = pos_weight
            weights[y == 0] = neg_weight
    
    # Apply PU weight for high_conf_clean pseudo-negatives
    if high_conf_clean_mask is not None:
        weights[high_conf_clean_mask] = pu_weight
    
    return weights


class EnsembleModel:
    """Ensemble of multiple model types."""
    
    def __init__(self, model_configs: list = None):
        """
        Args:
            model_configs: List of dicts with keys:
                - type: 'catboost', 'lightgbm', 'xgboost'
                - params: Model parameters (optional)
                - weight: Ensemble weight (optional)
        """
        if model_configs is None:
            model_configs = [
                {"type": "catboost", "weight": 0.4},
                {"type": "lightgbm", "weight": 0.35},
                {"type": "xgboost", "weight": 0.25},
            ]
        
        self.configs = model_configs
        self.models = {}
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            sample_weight: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Fit all models in the ensemble."""
        val_preds = {}
        
        for cfg in self.configs:
            model_type = cfg["type"]
            params = cfg.get("params")
            
            print(f"\n--- Training {model_type} ---")
            trainer = get_trainer(model_type)
            model, preds = trainer(X_train, y_train, X_val, y_val, params, sample_weight)
            
            self.models[model_type] = model
            val_preds[model_type] = preds
        
        return val_preds
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all models."""
        preds = {}
        for model_type, model in self.models.items():
            preds[model_type] = predict_with_model(model, X)
        return preds
    
    def predict_ensemble(self, X: np.ndarray, weights: dict = None) -> np.ndarray:
        """Get weighted ensemble prediction."""
        if weights is None:
            weights = {cfg["type"]: cfg.get("weight", 1.0) for cfg in self.configs}
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        preds = np.zeros(len(X))
        for model_type, w in weights.items():
            if model_type in self.models:
                preds += w * predict_with_model(self.models[model_type], X)
        
        return preds
    
    def save(self, dir_path: str):
        """Save ensemble to directory."""
        os.makedirs(dir_path, exist_ok=True)
        
        for model_type, model in self.models.items():
            path = os.path.join(dir_path, f"{model_type}.pkl")
            save_model(model, path)
        
        # Save config
        config_path = os.path.join(dir_path, "config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(self.configs, f)
    
    def load(self, dir_path: str):
        """Load ensemble from directory."""
        config_path = os.path.join(dir_path, "config.pkl")
        with open(config_path, "rb") as f:
            self.configs = pickle.load(f)
        
        for cfg in self.configs:
            model_type = cfg["type"]
            path = os.path.join(dir_path, f"{model_type}.pkl")
            if os.path.exists(path):
                self.models[model_type] = load_model(path)
