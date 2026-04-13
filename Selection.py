"""
Offline Cluster Number Selection Tool for Dual-mode Self-Organizing Map (Dual-SOM).

This standalone script provides an Exploratory Data Analysis (EDA) utility to
automatically evaluate and visualize different cluster numbers (k) based on
pre-trained SOM weights. It acts as an offline visual aid to help researchers
determine the optimal `n_clusters` parameter before executing the main
unsupervised clustering pipeline.

Usage:
    $ python evaluate_k.py --config params.json --k_min 2 --k_max 15
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

# Internal module imports for data handling and autoencoder
from preprocessing import get_dataset
from sparse_autoencoder import encode_decode, set_ae_args

class SOMClusterSelector:
    """
    Evaluates cluster compactness using angular distance to find the optimal K.

    This class reads unified settings from `params.json`, loads the serialized
    neural network weights, and computes the Delta L(k) metric to suggest the
    most natural topological boundaries in the data.

    Attributes:
        config_path (str): Filepath to the JSON configuration file.
        parameters (dict): Hyperparameter dictionary loaded from the JSON file.
        som_weights (np.ndarray): The loaded pre-trained SOM weight matrix.
        encoded_data (np.ndarray): The data features after SAE dimensionality reduction.
        k_range (list): The sequence of K values evaluated.
        delta_L_scores (list): The computed Delta L(k) scores corresponding to k_range.
        optimal_k (int): The recommended number of clusters (minimum Delta L).
    """

    def __init__(self, config_path='params.json'):
        """
        Initializes the selector and parses the configuration file.

        Args:
            config_path (str, optional): Path to the configuration JSON.
                                         Defaults to 'params.json'.
        """
        self.config_path = config_path
        self.parameters = self._read_parameters()
        self.som_weights = None
        self.encoded_data = None

        # Tracking metrics for evaluation and plotting
        self.k_range = []
        self.delta_L_scores = []
        self.optimal_k = None

    def _read_parameters(self):
        """
        Reads parameters from the JSON file.

        Unlike the main pipeline, this offline tool enforces strict existence
        of the configuration file. It assumes the model has already been trained.

        Returns:
            dict: The loaded configuration dictionary.

        Raises:
            SystemExit: If the configuration file is not found.
        """
        if not os.path.exists(self.config_path):
            print(f"[Error] Config '{self.config_path}' not found.")
            print("Please run main.py first to generate the configuration and train the model.")
            sys.exit(1)

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_data_and_model(self):
        """
        Encodes the training data and loads the pre-trained SOM weights.

        This method actively modifies the runtime parameters to FORCE the pipeline
        into 'load' mode, preventing accidental retraining during offline analysis.

        Returns:
            bool: True if data and model are loaded successfully, False otherwise.
        """
        # Force bypass training to exclusively perform offline analysis
        self.parameters['ae_load_model'] = True
        self.parameters['som_load_model'] = True

        # Phase 1: Read and encode raw data into the latent space
        train_data_path = self.parameters.get('train_data_path', 'Datas/MNIST/train_data.csv')
        if not os.path.exists(train_data_path):
            print(f"[Error] Training data not found at {train_data_path}")
            return False

        print(f"\n>>> Loading and encoding data from {train_data_path}...")
        train_data = get_dataset(train_data_path)
        set_ae_args(self.parameters)
        coded_data = encode_decode(train_data)

        self.encoded_data = coded_data[0] if isinstance(coded_data, tuple) else coded_data

        # Phase 2: Check and load the pre-trained SOM weight matrix
        model_path = self.parameters.get('som_model_path', 'weight/som_weights.npy')

        if not os.path.exists(model_path):
            print("\n" + "="*60)
            print(" [Error] Pre-trained SOM model not found!")
            print(f" Looked at path: {model_path}")
            print(" Solution: Please run main.py first to train and save the network.")
            print("="*60 + "\n")
            return False

        print(f">>> Loading pre-trained SOM model from {model_path}...")

        # Support for both standard NumPy binaries and legacy Pickle objects
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
            # Safely extract weights depending on the serialization structure
            self.som_weights = getattr(loaded_model, '_weights', None) or getattr(loaded_model.som, '_weights', None)
        elif model_path.endswith('.npy'):
            self.som_weights = np.load(model_path)
        else:
            print(f"[Error] Unsupported model file format '{model_path}'.")
            return False

        if self.som_weights is None:
            print("[Error] Could not extract weights from the loaded model.")
            return False

        print(f">>> Successfully loaded weights with shape: {self.som_weights.shape}")
        return True

    def _angular_distance_matrix(self, W1, W2):
        """
        Computes the pairwise angular distance matrix between two sets of vectors.

        Args:
            W1 (np.ndarray): First vector matrix of shape (N, D).
            W2 (np.ndarray): Second vector matrix of shape (M, D).

        Returns:
            np.ndarray: Matrix of shape (N, M) containing angular distances in radians.
        """
        W1_norm = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-8)
        W2_norm = W2 / (np.linalg.norm(W2, axis=1, keepdims=True) + 1e-8)
        sim = np.clip(np.dot(W1_norm, W2_norm.T), -1.0, 1.0)
        ang = np.arccos(sim) * 180 / np.pi
        ang = np.minimum(ang, 180 - ang)
        return ang / 180 * np.pi

    def _compute_angular_L_score(self, weights, labels, centroids):
        """
        Computes the L(k) dispersion metric based on angular distances.

        Args:
            weights (np.ndarray): Flattened array of SOM weights.
            labels (np.ndarray): Cluster assignments for each weight vector.
            centroids (np.ndarray): Learned cluster centroids.

        Returns:
            float: The aggregated L(k) score representing cluster compactness.
        """
        L_k = 0.0
        for i, centroid in enumerate(centroids):
            cluster_weights = weights[labels == i]
            N_ci = len(cluster_weights)

            if N_ci == 0:
                continue

            c_matrix = centroid.reshape(1, -1)
            dist_matrix = self._angular_distance_matrix(cluster_weights, c_matrix)
            L_k += np.sum(dist_matrix) / N_ci

        return L_k

    def evaluate(self, k_min=2, k_max=15):
        """
        Evaluates a range of cluster numbers to find the optimal K (Km).

        This method fits K-Means sequentially from k_min to k_max. It computes
        the difference in dispersion (Delta L) between successive K values.
        The minimum Delta L represents the point of diminishing topological returns.

        Args:
            k_min (int, optional): Minimum clusters to evaluate. Defaults to 2.
            k_max (int, optional): Maximum clusters to evaluate. Defaults to 15.

        Returns:
            tuple: (k_range list, delta_L_scores list, optimal_k integer).

        Raises:
            RuntimeError: If called before `load_data_and_model()`.
        """
        if self.som_weights is None:
            raise RuntimeError("Model not loaded. Call load_data_and_model() first.")

        if k_min <= 1:
            k_min = 2 # k_min - 1 must be at least 1 for the baseline L(k-1) calculation

        weights = self.som_weights
        if len(weights.shape) > 2:
            weights = weights.reshape(-1, weights.shape[-1])

        # L2 Normalize weights to align spherical K-Means with angular distance logic
        normalized_weights = normalize(weights, norm='l2')

        self.k_range = list(range(k_min, k_max + 1))
        self.delta_L_scores = []

        # Extract synchronized KMeans hyperparameters from JSON to match main pipeline
        max_iterations = self.parameters.get('kmeans_max_iter', 300)
        tol_threshold = self.parameters.get('kmeans_threshold', 1e-4)

        print(f"\n>>> Evaluating cluster numbers from {k_min} to {k_max}...")
        print(f">>> KMeans Config: max_iter={max_iterations}, tol={tol_threshold}")

        eval_range = list(range(k_min - 1, k_max + 1))
        temp_L_scores = {}

        # Stage 1: Compute absolute L(k) dispersion for all candidates
        for k in eval_range:
            kmeans = KMeans(
                n_clusters=k,
                max_iter=max_iterations,
                tol=tol_threshold,
                random_state=1,
                n_init=10
            )
            y_pred_weights = kmeans.fit_predict(normalized_weights)
            centroids = kmeans.cluster_centers_

            temp_L_scores[k] = self._compute_angular_L_score(weights, y_pred_weights, centroids)

        print("-" * 50)

        # Stage 2: Calculate Delta L(k) and identify the minimum point
        for k in self.k_range:
            L_current = temp_L_scores[k]
            L_previous = temp_L_scores[k - 1]
            delta_L = abs(L_current - L_previous)

            self.delta_L_scores.append(delta_L)
            print(f"k={k:02d} | L(k): {L_current:.4f} | Delta L(k): {delta_L:.4f}")

        print("-" * 50)

        best_idx = np.argmin(self.delta_L_scores)
        self.optimal_k = self.k_range[best_idx]
        print(f"\n>>> Recommended Optimal Cluster Number (Km): {self.optimal_k} (Minimum Delta L)")

        return self.k_range, self.delta_L_scores, self.optimal_k

    def plot_metrics(self):
        """
        Visualizes the Delta L(k) metric using Matplotlib.

        Renders a line plot identifying the elbow/minimum point, which acts
        as visual proof for selecting the recommended number of clusters.

        Raises:
            RuntimeError: If called before `evaluate()` populates the metrics.
        """
        if not self.k_range:
            raise RuntimeError("No evaluation data available. Call evaluate() before plotting.")

        fig, ax = plt.subplots(figsize=(8, 5))
        x_major_locator = MultipleLocator(1)

        # Plot the Delta L(k) trajectory
        ax.plot(self.k_range, self.delta_L_scores, color="#D32F2F", marker='s', linestyle='-', linewidth=2)
        ax.xaxis.set_major_locator(x_major_locator)

        # Highlight the optimal Km point on the graph
        opt_idx = self.k_range.index(self.optimal_k)
        ax.scatter([self.optimal_k], [self.delta_L_scores[opt_idx]],
                   color='#1976D2', s=120, zorder=5,
                   label=f'Optimal Km = {self.optimal_k}')

        # Formatting and Labels
        ax.legend(fontsize=11)
        ax.set_xlabel('Number of clusters (k)', fontsize=12)
        ax.set_ylabel(r'$\Delta L(k) = |L(k) - L(k-1)|$', fontsize=12)
        ax.set_title('Cluster Number Evaluation (Lower is better)', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

# =====================================================================
# CLI Execution Block
# =====================================================================
if __name__ == "__main__":
    # Parse command line arguments for flexible execution
    parser = argparse.ArgumentParser(description="Offline SOM Cluster Evaluation Tool")
    parser.add_argument('--config', type=str, default='params.json', help="Path to JSON config")
    parser.add_argument('--k_min', type=int, default=2, help="Minimum number of clusters to evaluate")
    parser.add_argument('--k_max', type=int, default=15, help="Maximum number of clusters to evaluate")
    args = parser.parse_args()

    # Instantiate the evaluation tool class
    selector = SOMClusterSelector(config_path=args.config)

    # Validate environment and model readiness
    if selector.load_data_and_model():

        # Run the core evaluation logic
        selector.evaluate(k_min=args.k_min, k_max=args.k_max)

        # Render the visual plot for user review
        print("\n>>> Plotting evaluation metrics. Close the window to exit.")
        selector.plot_metrics()

        # Provide actionable concluding instructions
        print(f"\n>>> Done! If you agree with the results, please update 'n_clusters': {selector.optimal_k}")
        print(f">>> in your '{args.config}' file before running the main pipeline in unsupervised mode.\n")
