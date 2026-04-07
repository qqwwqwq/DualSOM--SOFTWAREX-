"""
Cluster Number Selection Tool for Dual-mode Self-Organizing Map (Dual-SOM).

This module provides the `SOMClusterSelector` class to automatically evaluate
and visualize different cluster numbers (k) based on the exact angular distance
implementation provided in the project. It focuses strictly on minimizing Delta L(k).
"""

import argparse
import json
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Import pipeline components
from preprocessing import get_dataset
from sparse_autoencoder import encode_decode, set_ae_args

class SOMClusterSelector:
    def __init__(self, config_path='params.json'):
        """
        Initializes the selector by loading configurations.
        """
        self.config_path = config_path
        self.parameters = self._read_parameters()
        self.som_weights = None
        self.encoded_data = None

        # Lists to store evaluation metrics
        self.k_range = []
        self.delta_L_scores = []
        self.optimal_k = None

    def _read_parameters(self):
        """Reads parameters from the JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config '{self.config_path}' not found. Please run main.py first.")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_data_and_model(self):
        """
        Encodes the training data and loads the pre-trained SOM weights.
        Returns True if successful, False if the model is missing or fails to load.
        """
        # ==========================================
        # Force bypass training regardless of JSON configuration
        # ==========================================
        self.parameters['ae_load_model'] = True
        self.parameters['som_load_model'] = True

        # 1. Read and encode data
        train_data_path = self.parameters.get('train_data_path', 'data/train_data.csv')
        print(f"\n>>> Loading and encoding training data from {train_data_path}...")
        train_data = get_dataset(train_data_path)
        set_ae_args(self.parameters)
        coded_data = encode_decode(train_data)

        if isinstance(coded_data, tuple):
            self.encoded_data = coded_data[0]
        else:
            self.encoded_data = coded_data

        # 2. Check and Load pre-trained SOM weights
        model_path = self.parameters.get('som_model_path')

        if not model_path or not os.path.exists(model_path):
            print("\n" + "="*60)
            print(" [Error] Pre-trained SOM model not found!")
            print(f" Looked at path: {model_path}")
            print(" Solution: Please run main.py first to train the network.")
            print("="*60 + "\n")
            return False

        print(f"\n>>> Loading pre-trained SOM model from {model_path}...")
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
            self.som_weights = getattr(loaded_model, '_weights', None) or getattr(loaded_model.som, '_weights', None)
        elif model_path.endswith('.npy'):
            self.som_weights = np.load(model_path)
        else:
            print(f"\n [Error] Unsupported model file format '{model_path}'.")
            return False

        if self.som_weights is None:
            print("\n [Error] Could not extract weights from the loaded model.")
            return False

        print(f"Successfully loaded weights with shape: {self.som_weights.shape}")
        return True

    def _angular_distance_matrix(self, W1, W2):
        """
        Computes the pairwise angular distance matrix between two sets of vectors.
        (Exact implementation integrated from the user's codebase)
        """
        W1_norm = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-8)
        W2_norm = W2 / (np.linalg.norm(W2, axis=1, keepdims=True) + 1e-8)
        sim = np.clip(np.dot(W1_norm, W2_norm.T), -1.0, 1.0)
        ang = np.arccos(sim) * 180 / np.pi
        ang = np.minimum(ang, 180 - ang)
        return ang / 180 * np.pi

    def _compute_angular_L_score(self, weights, labels, centroids):
        """
        Computes the L(k) metric based on the custom angular distances.
        L(k) = sum( (1/N_ci) * sum(angular_distance(w_ci, c_i)) )
        """
        L_k = 0.0

        for i, centroid in enumerate(centroids):
            cluster_weights = weights[labels == i]
            N_ci = len(cluster_weights)

            if N_ci == 0:
                continue

            # Reshape centroid to (1, D) to act as W2 in the distance matrix function
            c_matrix = centroid.reshape(1, -1)

            # Calculate angular distances between all weights in the cluster and the centroid
            dist_matrix = self._angular_distance_matrix(cluster_weights, c_matrix)

            # Add the average angular distance of this cluster to L(k)
            L_k += np.sum(dist_matrix) / N_ci

        return L_k

    def evaluate(self, k_min=10, k_max=25):
        """
        Evaluates cluster numbers by finding the minimum of Delta L(k).
        """
        if self.som_weights is None:
            raise RuntimeError("Model not loaded. Call load_data_and_model() first.")

        if k_min <= 1:
            k_min = 2 # k_min - 1 must be at least 1 for clustering L(k) math

        # Flatten SOM weights to 2D matrix
        weights = self.som_weights
        if len(weights.shape) > 2:
            weights = weights.reshape(-1, weights.shape[-1])

        # Normalize weights for KMeans to approximate spherical clustering
        normalized_weights = normalize(weights, norm='l2')

        self.k_range = list(range(k_min, k_max + 1))
        self.delta_L_scores = []

        print(f"\n>>> Evaluating cluster numbers from {k_min} to {k_max}...")

        # We need to compute L(k_min - 1) to calculate the first Delta L
        eval_range = list(range(k_min - 1, k_max + 1))
        temp_L_scores = {}

        for k in eval_range:
            kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=1, n_init=10)
            y_pred_weights = kmeans.fit_predict(normalized_weights)
            centroids = kmeans.cluster_centers_

            # Calculate L(k) (Required for Delta L calculation)
            L_k = self._compute_angular_L_score(weights, y_pred_weights, centroids)
            temp_L_scores[k] = L_k

        # Calculate Delta L(k) = | L(k) - L(k-1) |
        for k in self.k_range:
            L_current = temp_L_scores[k]
            L_previous = temp_L_scores[k - 1]
            delta_L = abs(L_current - L_previous)

            self.delta_L_scores.append(delta_L)

            print(f"k={k:02d} | L(k): {L_current:.4f} | Delta L(k) = |L({k})-L({k-1})|: {delta_L:.4f}")

        # Identify the optimal k (Km) that minimizes Delta L(k)
        best_idx = np.argmin(self.delta_L_scores)
        self.optimal_k = self.k_range[best_idx]
        print(f"\n>>> Recommended Optimal Cluster Number (Km): {self.optimal_k} (Minimum Delta L)")

        return self.k_range, self.delta_L_scores, self.optimal_k

    def plot_metrics(self):
        """
        Visualizes only the Delta L(k) metric to demonstrate the optimal cluster selection.
        """
        if not self.k_range:
            raise RuntimeError("No evaluation data available. Call evaluate() before plotting.")

        fig, ax = plt.subplots(figsize=(8, 5))
        x_major_locator = MultipleLocator(1)

        # Plot Delta L(k)
        ax.plot(self.k_range, self.delta_L_scores, color="red", marker='s', linestyle='-')
        ax.xaxis.set_major_locator(x_major_locator)

        # Highlight the minimum Delta L(k)
        opt_idx = self.k_range.index(self.optimal_k)
        ax.scatter([self.optimal_k], [self.delta_L_scores[opt_idx]], color='black', s=100, zorder=5, label=f'Optimal Km = {self.optimal_k}')
        ax.legend()

        ax.set_xlabel('Number of clusters (k)', fontsize=12)
        ax.set_ylabel('ΔL(k) = |L(k) - L(k-1)|', fontsize=12)
        ax.set_title('Absolute Difference ΔL(k) (Lower is better)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

# =====================================================================
# CLI Execution Block
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOM Cluster Number Selection Tool")
    parser.add_argument('--config', type=str, default='params.json', help="Path to JSON config")
    parser.add_argument('--k_min', type=int, default=1, help="Minimum number of clusters")
    parser.add_argument('--k_max', type=int, default=11, help="Maximum number of clusters")
    args = parser.parse_args()

    # Instantiate the tool class
    selector = SOMClusterSelector(config_path=args.config)

    # Check if data and model loaded successfully
    if not selector.load_data_and_model():
        sys.exit(1)

    # Run the evaluation
    selector.evaluate(k_min=args.k_min, k_max=args.k_max)

    print("\n>>> Plotting evaluation metrics. Close the window to exit.")
    selector.plot_metrics()

    print(f"\n>>> Done. Please update 'n_clusters' to {selector.optimal_k} in {args.config} before running the unsupervised mode.")