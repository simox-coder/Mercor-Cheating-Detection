"""Score propagation on graph to enhance predictions."""
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Optional
from tqdm import tqdm

from .config import SEED


def build_adjacency_list(edges_df: pd.DataFrame) -> Dict[str, list]:
    """Build adjacency list from edge DataFrame."""
    adj_list = defaultdict(list)
    for _, row in edges_df.iterrows():
        adj_list[row["user_a"]].append(row["user_b"])
        adj_list[row["user_b"]].append(row["user_a"])
    return dict(adj_list)


def simple_diffusion(predictions: pd.DataFrame, 
                     adj_list: Dict[str, list],
                     alpha: float = 0.3,
                     n_iterations: int = 3,
                     user_col: str = "user_hash",
                     pred_col: str = "prediction") -> np.ndarray:
    """
    Simple graph diffusion: new_score = (1-alpha)*score + alpha*mean(neighbor_scores)
    
    Args:
        predictions: DataFrame with user_hash and prediction columns
        adj_list: Adjacency list
        alpha: Diffusion weight (0 = no diffusion, 1 = full neighbor averaging)
        n_iterations: Number of diffusion iterations
        user_col: User ID column name
        pred_col: Prediction column name
    
    Returns:
        Propagated predictions array
    """
    # Create score lookup
    score_map = dict(zip(predictions[user_col], predictions[pred_col]))
    
    for iteration in range(n_iterations):
        new_scores = {}
        
        for user, score in score_map.items():
            if user in adj_list and len(adj_list[user]) > 0:
                # Get neighbor scores
                neighbor_scores = [
                    score_map.get(n, score) for n in adj_list[user]
                ]
                neighbor_mean = np.mean(neighbor_scores)
                
                # Diffuse
                new_scores[user] = (1 - alpha) * score + alpha * neighbor_mean
            else:
                new_scores[user] = score
        
        score_map = new_scores
    
    # Return in original order
    return np.array([score_map.get(u, predictions.loc[predictions[user_col] == u, pred_col].values[0])
                     for u in predictions[user_col]])


def pagerank_diffusion(predictions: pd.DataFrame,
                       adj_list: Dict[str, list],
                       alpha: float = 0.85,
                       n_iterations: int = 10,
                       user_col: str = "user_hash",
                       pred_col: str = "prediction") -> np.ndarray:
    """
    PageRank-style diffusion with personalization.
    
    Uses the predicted scores as personalization vector.
    """
    users = list(predictions[user_col])
    n = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}
    
    # Initial scores from predictions
    scores = predictions[pred_col].values.copy()
    personalization = scores.copy()
    
    for _ in range(n_iterations):
        new_scores = np.zeros(n)
        
        for i, user in enumerate(users):
            if user in adj_list and len(adj_list[user]) > 0:
                neighbors = adj_list[user]
                neighbor_sum = 0.0
                count = 0
                
                for n in neighbors:
                    if n in user_to_idx:
                        neighbor_sum += scores[user_to_idx[n]]
                        count += 1
                
                if count > 0:
                    new_scores[i] = alpha * (neighbor_sum / count) + (1 - alpha) * personalization[i]
                else:
                    new_scores[i] = personalization[i]
            else:
                new_scores[i] = personalization[i]
        
        scores = new_scores
    
    return scores


def label_propagation(predictions: pd.DataFrame,
                      adj_list: Dict[str, list],
                      known_labels: Dict[str, int] = None,
                      alpha: float = 0.5,
                      n_iterations: int = 5,
                      user_col: str = "user_hash",
                      pred_col: str = "prediction") -> np.ndarray:
    """
    Label propagation that incorporates known labels.
    
    IMPORTANT: Only use OOF predictions to avoid leakage!
    """
    # Create score lookup
    score_map = dict(zip(predictions[user_col], predictions[pred_col]))
    
    # Apply known labels as strong anchors
    if known_labels:
        for user, label in known_labels.items():
            if user in score_map:
                score_map[user] = float(label)
    
    for _ in range(n_iterations):
        new_scores = {}
        
        for user in score_map:
            # Skip known labels (keep them fixed)
            if known_labels and user in known_labels:
                new_scores[user] = float(known_labels[user])
                continue
            
            score = score_map[user]
            
            if user in adj_list and len(adj_list[user]) > 0:
                neighbor_scores = [
                    score_map.get(n, 0.5) for n in adj_list[user]
                ]
                neighbor_mean = np.mean(neighbor_scores)
                new_scores[user] = (1 - alpha) * score + alpha * neighbor_mean
            else:
                new_scores[user] = score
        
        score_map = new_scores
    
    # Return in original order
    return np.array([score_map[u] for u in predictions[user_col]])


def apply_propagation(predictions_df: pd.DataFrame,
                      edges_df: pd.DataFrame,
                      method: str = "simple",
                      **kwargs) -> np.ndarray:
    """
    Apply score propagation to predictions.
    
    Args:
        predictions_df: DataFrame with user_hash and prediction columns
        edges_df: DataFrame with user_a and user_b columns
        method: 'simple', 'pagerank', or 'label_prop'
        **kwargs: Additional arguments for the propagation method
    
    Returns:
        Propagated predictions array
    """
    adj_list = build_adjacency_list(edges_df)
    
    if method == "simple":
        return simple_diffusion(predictions_df, adj_list, **kwargs)
    elif method == "pagerank":
        return pagerank_diffusion(predictions_df, adj_list, **kwargs)
    elif method == "label_prop":
        return label_propagation(predictions_df, adj_list, **kwargs)
    else:
        raise ValueError(f"Unknown propagation method: {method}")


def propagate_and_blend(base_preds: np.ndarray,
                        user_hashes: np.ndarray,
                        edges_df: pd.DataFrame,
                        prop_weight: float = 0.3,
                        method: str = "simple",
                        **kwargs) -> np.ndarray:
    """
    Propagate predictions and blend with original.
    
    Args:
        base_preds: Original predictions
        user_hashes: User hash array
        edges_df: Graph edges
        prop_weight: Weight for propagated predictions
        method: Propagation method
    
    Returns:
        Blended predictions
    """
    pred_df = pd.DataFrame({
        "user_hash": user_hashes,
        "prediction": base_preds
    })
    
    prop_preds = apply_propagation(pred_df, edges_df, method, **kwargs)
    
    # Blend
    blended = (1 - prop_weight) * base_preds + prop_weight * prop_preds
    
    # Clip to valid range
    return np.clip(blended, 0, 1)
