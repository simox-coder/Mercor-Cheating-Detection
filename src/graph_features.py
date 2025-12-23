"""Graph feature engineering: degree, PageRank, components, node2vec embeddings."""
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, Tuple, Optional
import os
import pickle
from tqdm import tqdm

from .config import (
    NODE2VEC_DIM, NODE2VEC_WALK_LENGTH, NODE2VEC_NUM_WALKS,
    NODE2VEC_P, NODE2VEC_Q, NODE2VEC_WORKERS, SEED
)


class UnionFind:
    """Union-Find data structure for connected components."""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def compute_connected_components(edges_df: pd.DataFrame) -> Dict[str, int]:
    """Compute connected components using Union-Find."""
    print("Computing connected components...")
    uf = UnionFind()
    
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Union-Find"):
        uf.union(row["user_a"], row["user_b"])
    
    # Assign component IDs
    component_map = {}
    comp_id = 0
    root_to_id = {}
    
    for node in tqdm(list(uf.parent.keys()), desc="Assigning components"):
        root = uf.find(node)
        if root not in root_to_id:
            root_to_id[root] = comp_id
            comp_id += 1
        component_map[node] = root_to_id[root]
    
    print(f"Found {comp_id} connected components")
    return component_map


def compute_degree_features(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Compute node degree features."""
    print("Computing degree features...")
    
    # Count degrees
    degree_a = edges_df.groupby("user_a").size().reset_index(name="degree")
    degree_a.columns = ["user_hash", "degree"]
    
    degree_b = edges_df.groupby("user_b").size().reset_index(name="degree")
    degree_b.columns = ["user_hash", "degree"]
    
    # Combine
    degree_df = pd.concat([degree_a, degree_b])
    degree_df = degree_df.groupby("user_hash")["degree"].sum().reset_index()
    
    # Add log degree
    degree_df["log_degree"] = np.log1p(degree_df["degree"])
    
    print(f"Computed degrees for {len(degree_df)} nodes")
    return degree_df


def compute_component_size(edges_df: pd.DataFrame, component_map: Dict[str, int]) -> pd.DataFrame:
    """Compute component size for each node."""
    print("Computing component sizes...")
    
    # Count nodes per component
    comp_sizes = defaultdict(int)
    for node, comp_id in component_map.items():
        comp_sizes[comp_id] += 1
    
    # Create dataframe
    data = []
    for node, comp_id in component_map.items():
        data.append({
            "user_hash": node,
            "component_id": comp_id,
            "component_size": comp_sizes[comp_id],
            "log_component_size": np.log1p(comp_sizes[comp_id])
        })
    
    return pd.DataFrame(data)


def compute_pagerank(edges_df: pd.DataFrame, max_nodes: int = 500000) -> pd.DataFrame:
    """Compute PageRank for graph nodes."""
    print("Computing PageRank...")
    
    # Build graph
    G = nx.Graph()
    
    # Get unique nodes
    all_nodes = set(edges_df["user_a"].unique()) | set(edges_df["user_b"].unique())
    
    if len(all_nodes) > max_nodes:
        print(f"Graph too large ({len(all_nodes)} nodes), using approximate PageRank")
        # Use degree as proxy for PageRank on large graphs
        degree_df = compute_degree_features(edges_df)
        total_degree = degree_df["degree"].sum()
        degree_df["pagerank"] = degree_df["degree"] / total_degree
        return degree_df[["user_hash", "pagerank"]]
    
    # Add edges
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Building graph"):
        G.add_edge(row["user_a"], row["user_b"])
    
    # Compute PageRank
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    
    pr_df = pd.DataFrame([
        {"user_hash": node, "pagerank": score}
        for node, score in pr.items()
    ])
    
    print(f"Computed PageRank for {len(pr_df)} nodes")
    return pr_df


def random_walk(adj_list: Dict, start_node: str, walk_length: int, p: float, q: float) -> list:
    """Perform a biased random walk."""
    walk = [start_node]
    
    if start_node not in adj_list or len(adj_list[start_node]) == 0:
        return walk
    
    walk.append(np.random.choice(adj_list[start_node]))
    
    while len(walk) < walk_length:
        cur = walk[-1]
        prev = walk[-2]
        
        if cur not in adj_list or len(adj_list[cur]) == 0:
            break
        
        neighbors = adj_list[cur]
        probs = []
        
        for neighbor in neighbors:
            if neighbor == prev:
                probs.append(1.0 / p)
            elif neighbor in adj_list.get(prev, []):
                probs.append(1.0)
            else:
                probs.append(1.0 / q)
        
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        next_node = np.random.choice(neighbors, p=probs)
        walk.append(next_node)
    
    return walk


def generate_walks(adj_list: Dict, nodes: list, num_walks: int, walk_length: int,
                   p: float = 1.0, q: float = 1.0) -> list:
    """Generate random walks for all nodes."""
    walks = []
    
    for _ in tqdm(range(num_walks), desc="Generating walks"):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = random_walk(adj_list, node, walk_length, p, q)
            walks.append(walk)
    
    return walks


def compute_node2vec_embeddings(edges_df: pd.DataFrame, 
                                 dim: int = NODE2VEC_DIM,
                                 walk_length: int = NODE2VEC_WALK_LENGTH,
                                 num_walks: int = NODE2VEC_NUM_WALKS,
                                 p: float = NODE2VEC_P,
                                 q: float = NODE2VEC_Q,
                                 cache_path: str = None) -> pd.DataFrame:
    """Compute Node2Vec embeddings using gensim Word2Vec."""
    from gensim.models import Word2Vec
    
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return pd.read_pickle(cache_path)
    
    print("Computing Node2Vec embeddings...")
    np.random.seed(SEED)
    
    # Build adjacency list
    adj_list = defaultdict(list)
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Building adjacency"):
        adj_list[row["user_a"]].append(row["user_b"])
        adj_list[row["user_b"]].append(row["user_a"])
    
    nodes = list(adj_list.keys())
    print(f"Graph has {len(nodes)} nodes")
    
    # Generate walks
    walks = generate_walks(adj_list, nodes, num_walks, walk_length, p, q)
    print(f"Generated {len(walks)} walks")
    
    # Train Word2Vec
    print("Training Word2Vec...")
    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=5,
        min_count=1,
        sg=1,  # Skip-gram
        workers=NODE2VEC_WORKERS,
        seed=SEED,
        epochs=5
    )
    
    # Extract embeddings
    emb_cols = [f"node2vec_{i}" for i in range(dim)]
    data = []
    
    for node in tqdm(nodes, desc="Extracting embeddings"):
        if node in model.wv:
            emb = model.wv[node]
            data.append({"user_hash": node, **{col: emb[i] for i, col in enumerate(emb_cols)}})
    
    emb_df = pd.DataFrame(data)
    
    if cache_path:
        emb_df.to_pickle(cache_path)
        print(f"Saved embeddings to {cache_path}")
    
    print(f"Computed embeddings for {len(emb_df)} nodes")
    return emb_df


def compute_neighbor_stats(edges_df: pd.DataFrame, user_features: pd.DataFrame,
                           feature_cols: list) -> pd.DataFrame:
    """Compute neighbor aggregation statistics (mean, std, min, max)."""
    print("Computing neighbor statistics...")
    
    # Build adjacency list
    adj_list = defaultdict(list)
    for _, row in edges_df.iterrows():
        adj_list[row["user_a"]].append(row["user_b"])
        adj_list[row["user_b"]].append(row["user_a"])
    
    # Create feature lookup
    feature_lookup = user_features.set_index("user_hash")[feature_cols].to_dict("index")
    
    results = []
    for user in tqdm(adj_list.keys(), desc="Computing neighbor stats"):
        neighbors = adj_list[user]
        neighbor_feats = []
        
        for n in neighbors:
            if n in feature_lookup:
                neighbor_feats.append(feature_lookup[n])
        
        if len(neighbor_feats) == 0:
            continue
        
        neighbor_df = pd.DataFrame(neighbor_feats)
        
        row = {"user_hash": user}
        for col in feature_cols:
            if col in neighbor_df.columns:
                row[f"neighbor_mean_{col}"] = neighbor_df[col].mean()
                row[f"neighbor_std_{col}"] = neighbor_df[col].std()
        
        results.append(row)
    
    return pd.DataFrame(results)


def build_all_graph_features(edges_df: pd.DataFrame, 
                              cache_dir: str = None,
                              compute_embeddings: bool = True) -> pd.DataFrame:
    """Build all graph features."""
    
    # Degree features
    degree_df = compute_degree_features(edges_df)
    
    # Connected components
    component_map = compute_connected_components(edges_df)
    comp_df = compute_component_size(edges_df, component_map)
    
    # Merge degree and component features
    graph_feats = degree_df.merge(comp_df, on="user_hash", how="outer")
    
    # PageRank (approximate for large graphs)
    pr_df = compute_pagerank(edges_df)
    graph_feats = graph_feats.merge(pr_df, on="user_hash", how="outer")
    
    # Node2Vec embeddings (optional, computationally expensive)
    if compute_embeddings:
        cache_path = os.path.join(cache_dir, "node2vec_embeddings.pkl") if cache_dir else None
        emb_df = compute_node2vec_embeddings(edges_df, cache_path=cache_path)
        graph_feats = graph_feats.merge(emb_df, on="user_hash", how="outer")
    
    # Fill NaN for nodes not in graph
    graph_feats = graph_feats.fillna(0)
    
    return graph_feats, component_map
