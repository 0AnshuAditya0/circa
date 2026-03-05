import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from pipeline.config import ModelConfig
@dataclass
class ClusterMapping:
    node_id: int
    node_name: str
    confidence: float
    top_latent_dims: List[int]
class FeatureClusterMapper:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.n_clusters = config.n_causal_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.is_fitted = False
    def fit(self, latent_vectors: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.detach().cpu().numpy()
        if latent_vectors.ndim == 1:
            latent_vectors = latent_vectors.reshape(1, -1)
        self.kmeans.fit(latent_vectors)
        self.is_fitted = True
    def map_to_node(self, z: torch.Tensor) -> int:
        if not self.is_fitted:
            raise RuntimeError('FeatureClusterMapper must be fitted prior to mapping calls.')
        z_np = z.detach().cpu().numpy()
        if z_np.ndim == 1:
            z_np = z_np.reshape(1, -1)
        cluster_id = self.kmeans.predict(z_np)[0]
        return int(cluster_id)
    def map_to_node_probs(self, z: torch.Tensor) -> Dict[int, float]:
        if not self.is_fitted:
            raise RuntimeError('FeatureClusterMapper must be fitted prior to probability mapping.')
        z_np = z.detach().cpu().numpy()
        if z_np.ndim == 1:
            z_np = z_np.reshape(1, -1)
        distances = self.kmeans.transform(z_np)[0]
        epsilon = 1e-08
        similarities = 1.0 / (distances + epsilon)
        probs = similarities / np.sum(similarities)
        return {int(i): float(prob) for i, prob in enumerate(probs)}
    def get_cluster_mapping(self, z: torch.Tensor) -> ClusterMapping:
        probs = self.map_to_node_probs(z)
        node_id = max(probs, key=probs.get)
        confidence = probs[node_id]
        centroid = self.kmeans.cluster_centers_[node_id]
        top_dims = np.argsort(np.abs(centroid))[-3:][::-1].tolist()
        return ClusterMapping(node_id=node_id, node_name=self.get_node_name(node_id), confidence=confidence, top_latent_dims=top_dims)
    def get_node_name(self, node_id: int) -> str:
        return f'causal_node_{node_id}'
    def visualize_clusters(self, latent_vectors: Union[np.ndarray, torch.Tensor]=None) -> None:
        if not self.is_fitted:
            raise RuntimeError('Cannot visualize an unfitted network.')
        if latent_vectors is None:
            raise ValueError('Must supply evaluation data to visualize against fitted clusters.')
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.detach().cpu().numpy()
        labels = self.kmeans.predict(latent_vectors)
        if latent_vectors.shape[1] > 2:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                plot_data = pca.fit_transform(latent_vectors)
                centroids = pca.transform(self.kmeans.cluster_centers_)
                x_label, y_label = ('Principal Component 1', 'Principal Component 2')
            except ImportError:
                plot_data = latent_vectors[:, :2]
                centroids = self.kmeans.cluster_centers_[:, :2]
                x_label, y_label = ('Latent Feature 0', 'Latent Feature 1')
        else:
            plot_data = latent_vectors
            centroids = self.kmeans.cluster_centers_
            x_label, y_label = ('Latent Feature 0', 'Latent Feature 1')
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=labels, cmap='tab20', alpha=0.6, s=20)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Causal Nodes (Centroids)', edgecolors='white', linewidth=1.5)
        plt.title(f'CIRCA: DAG Causal Node Embeddings (K={self.n_clusters})')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.colorbar(scatter, label='Assigned Causal Node ID')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    def save(self, path: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError('Cluster map state not found, mapping layer unfitted.')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.kmeans, f)
    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f'Feature cluster mapping could not find binary at {path}')
        with open(path, 'rb') as f:
            self.kmeans = pickle.load(f)
        self.is_fitted = True