"""
Dual-mode Self-Organizing Map (Dual-SOM) Execution Pipeline.

This script serves as the primary entry point for training and evaluating the
Dual-SOM model. It seamlessly bridges deep representation learning (via Sparse
Autoencoders) with topological clustering and classification (via SOMs).

Key Features:
    - Dual Operational Modes: Supports both 'supervised' (classification) and
      'unsupervised' (clustering) learning without structural changes.
    - Auto-K Selection: Dynamically determines the optimal number of clusters
      using a custom angular distance dispersion metric (Delta L).
    - Safe Configuration Management: Merges user-defined JSON settings with
      immutable algorithm intrinsics to prevent accidental destabilization.

Usage:
    $ python main.py --config params.json
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Internal module imports for data handling, autoencoder, and the SOM model
from preprocessing import get_dataset
from sparse_autoencoder import encode_decode, set_ae_args
from Dualmap import DualSOM

# =====================================================================
# Group 1: User-Configurable Parameters (Exported to params.json)
# =====================================================================
# These parameters represent high-level workflow choices and tuning knobs
# specific to the user's dataset and hardware constraints.
USER_CONFIG_DEFAULTS = {
    # Workflow Selection
    "dataset_name": "mnist",          # Target dataset [Values: 'wut', 'pku', 'mnist', etc.]
    "run_mode": "supervised",         # Execution mode [Values: 'supervised', 'unsupervised']
    "device": "cuda",                 # Hardware acceleration target [Values: 'cuda', 'cpu']

    # Data Paths (Relative to the execution directory)
    "train_data_path": "Datas/MNIST/train_data.csv",
    "test_data_path": "Datas/MNIST/test_data.csv",

    # Model Saving/Loading Switches
    "som_load_model": False,          # If True, bypass SOM training and load pre-trained weights [Values: True, False]
    "som_model_path": "weight/som_weights.npy",
    "ae_load_model": False,           # If True, bypass SAE training and load pre-trained weights [Values: True, False]
    "ae_model_path": "weight/sparse_ae.pth",

    # Clustering Configurations (Active only when run_mode == 'unsupervised')
    "auto_find_clusters": False,      # If True, trigger SOMClusterSelector to find optimal K [Values: True, False]
    "k_min": 2,                       # Lower bound for auto-K search space [Range: > 1, Suggested: 2]
    "k_max": 12,                      # Upper bound for auto-K search space [Range: > k_min]
    "n_clusters": 10,                 # Hardcoded K, only needed when auto_find_clusters is false [Manually provided by the user]

    # Key Training Hyperparameters
    "ae_epochs": 150,                 # Number of training epochs for the SAE [Range: 50 - 500, Suggested: 150]
    "som_epochs": 50,                 # Number of global passes through the dataset for the SOM [e.g. 50, 100, 200...]
    "activation_distance": "angular",  # BMU distance metric [Values: 'angular' (directional/skeletal), 'euclidean' (general), 'cosine' (high-dim sparse)]
}

# =====================================================================
# Group 2: Intrinsic Algorithm Parameters (Internal, NOT in params.json)
# =====================================================================
# These constants govern the fundamental mathematical behavior of the Dual-SOM.
# They are hidden from the user config to maintain structural stability.
INTRINSIC_PARAMETERS = {
    # SOM Topological Hyperparameters
    "som_size_index": 10.0,           # Scale factor for the grid size heuristic (S ≈ sqrt(som_size_index * sqrt(N))) [Range: 1.0 - 20.0, Suggested: 10.0]
    "som_sigma": 4.0,                 # Initial neighborhood radius for lateral inhibition [Range: 1.0 - 10.0, Suggested: 4.0]
    "som_sigma_target": 0.01,         # Asymptotic lower bound for radius decay [Range: 0.001 - 0.1, Suggested: 0.01]
    "som_lr": 0.1,                    # Initial learning rate for Hebbian weight updates [Range: 0.01 - 1.0, Suggested: 0.1 - 0.5]
    "som_lr_target": 0.001,           # Asymptotic lower bound for learning rate decay [Range: 0.0001 - 0.01, Suggested: 0.001]
    "som_enable_validation": 1,       # Boolean flag for periodic terminal logging [Values: 1 (True) or 0 (False)]

    # K-Means Convergence Hyperparameters
    "kmeans_max_iter": 100,           # Max iterations before early stopping [Range: 100 - 1000, Suggested: 100 - 300]
    "kmeans_threshold": 1e-4,         # Centroid shift tolerance for convergence [Range: 1e-5 - 1e-2, Suggested: 1e-4]

    # Sparse Autoencoder (SAE) Optimization Hyperparameters
    "ae_batch_size": 32,              # Mini-batch size for SAE gradient descent [Values: 16, 32, 64, 128, 256. Suggested: 32 or 64]
    "ae_lr": 0.001,                   # Base learning rate for SAE Adam optimizer [Range: 1e-4 - 1e-2, Suggested: 0.001]
    "ae_reg_param": 0.001,            # L1 sparsity penalty coefficient (rho) [Range: 1e-5 - 1e-1, Suggested: 0.001]
}
# =====================================================================
# Integrated Cluster Selection Logic
# =====================================================================
class SOMClusterSelector:
    """
    Evaluates and automatically selects the optimal number of clusters (K)
    based on the trained SOM topological weights.

    This class implements the angular distance-based L(k) evaluation metric.
    It seeks the optimal cluster count (Km) by finding the minimum of the
    absolute difference between consecutive L scores (Delta L).

    Attributes:
        som_weights (np.ndarray): The trained SOM weight matrix.
    """

    def __init__(self, som_weights):
        """
        Initializes the selector.

        Args:
            som_weights (np.ndarray): Array of shape (n_neurons, features) or
                                      (grid_width, grid_height, features).
        """
        self.som_weights = som_weights

    def _angular_distance_matrix(self, W1, W2):
        """
        Computes the pairwise angular distance matrix between two sets of vectors.

        Angular distance is mathematically superior for directional/skeletal data.
        It calculates Cosine Similarity, clips it to [-1, 1] to prevent NaN errors
        during arccosine projection, and converts the angle to radians.

        Args:
            W1 (np.ndarray): First matrix of shape (N, D).
            W2 (np.ndarray): Second matrix of shape (M, D).

        Returns:
            np.ndarray: A distance matrix of shape (N, M) containing angular distances.
        """
        W1_norm = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-8)
        W2_norm = W2 / (np.linalg.norm(W2, axis=1, keepdims=True) + 1e-8)

        sim = np.clip(np.dot(W1_norm, W2_norm.T), -1.0, 1.0)
        ang = np.arccos(sim) * 180 / np.pi
        ang = np.minimum(ang, 180 - ang)
        return ang / 180 * np.pi

    def _compute_angular_L_score(self, weights, labels, centroids):
        """
        Calculates the internal clustering evaluation metric L(k).

        L(k) measures the average angular deviation of weights from their respective
        cluster centroids. Lower L(k) indicates tighter, more cohesive clusters.

        Args:
            weights (np.ndarray): The flattened SOM weight matrix.
            labels (np.ndarray): The cluster assignments for each weight.
            centroids (np.ndarray): The geometric centers of each cluster.

        Returns:
            float: The computed L(k) score.
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
        Executes the evaluation loop to find the optimal K.

        It iteratively clusters the SOM weights using K-Means for k in [k_min-1, k_max],
        computes L(k) for each, and then calculates Delta L(k) = |L(k) - L(k-1)|.
        The optimal K corresponds to the minimum Delta L(k).

        Args:
            k_min (int, optional): Minimum cluster boundary. Defaults to 2.
            k_max (int, optional): Maximum cluster boundary. Defaults to 15.

        Returns:
            int: The dynamically selected optimal number of clusters (Km).
        """
        if k_min <= 1:
            k_min = 2

        weights = self.som_weights

        # Flatten the SOM grid from 3D to 2D matrix if necessary
        if len(weights.shape) > 2:
            weights = weights.reshape(-1, weights.shape[-1])

        # L2 Normalization ensures spherical K-Means aligns with the angular metric
        normalized_weights = normalize(weights, norm='l2')
        k_range = list(range(k_min, k_max + 1))
        delta_L_scores = []

        print(f"\n>>> Auto-Evaluating optimal clusters from K={k_min} to {k_max}...")

        eval_range = list(range(k_min - 1, k_max + 1))
        temp_L_scores = {}

        # Compute L(k) for all candidate sizes
        for k in eval_range:
            kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=1, n_init=10)
            y_pred_weights = kmeans.fit_predict(normalized_weights)
            centroids = kmeans.cluster_centers_
            temp_L_scores[k] = self._compute_angular_L_score(weights, y_pred_weights, centroids)

        # Compute Delta L(k) and identify the minimum
        for k in k_range:
            L_current = temp_L_scores[k]
            L_previous = temp_L_scores[k - 1]
            delta_L = abs(L_current - L_previous)
            delta_L_scores.append(delta_L)
            print(f"    k={k:02d} | L(k): {L_current:.4f} | Delta L(k): {delta_L:.4f}")

        best_idx = np.argmin(delta_L_scores)
        optimal_k = k_range[best_idx]
        print(f">>> Recommended Optimal Cluster Number (Km): {optimal_k} (Minimum Delta L)")

        return optimal_k


# =====================================================================
# Main Pipeline Helpers
# =====================================================================
def create_default_params(json_path):
    """
    Initializes the configuration file for first-time setup.

    Generates a fresh JSON file containing ONLY the `USER_CONFIG_DEFAULTS`
    to prevent users from directly modifying sensitive intrinsic variables.

    Args:
        json_path (str): Filepath for the generated JSON config.
    """
    if not os.path.exists(json_path):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(USER_CONFIG_DEFAULTS, f, indent=4)
        print(f"Created user-configurable parameter file at {json_path}")

def read_parameters(json_path):
    """
    Loads user configurations and performs a safe-merge with intrinsic parameters.

    This strategy guarantees that downstream models receive a complete parameter
    dictionary, falling back to algorithm defaults if keys are missing from the JSON.

    Args:
        json_path (str): Filepath to read the JSON config from.

    Returns:
        dict: A fully populated dictionary of all model hyper-parameters.
    """
    merged_params = INTRINSIC_PARAMETERS.copy()

    if not os.path.exists(json_path):
        print(f"Warning: Config '{json_path}' not found. Using standard defaults.")
        merged_params.update(USER_CONFIG_DEFAULTS)
    else:
        with open(json_path, 'r', encoding='utf-8') as f:
            user_params = json.load(f)
            merged_params.update(user_params)

    return merged_params

def evaluate_and_print(y_true, y_pred, mode, dataset, stage_label):
    """
    Standardized evaluation logging module.

    Dynamically computes performance metrics based on the operation mode
    (Accuracy/F1 for classification, NMI/AMI/Homogeneity for clustering).

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels or cluster assignments.
        mode (str): Execution mode ('supervised' or 'unsupervised').
        dataset (str): Dataset identifier for terminal display.
        stage_label (str): Pipeline stage indicator (e.g., "TRAINING" or "TESTING").
    """
    print("\n" + "=" * 45)
    print(f"Metrics Summary - {stage_label} ({dataset.upper()} - {mode.upper()})")
    print("=" * 45)

    if mode == 'supervised':
        print("Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        metrics_dict = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0)
        }
    else:
        metrics_dict = {
            "nmi": metrics.normalized_mutual_info_score(y_true, y_pred),
            "ami": metrics.adjusted_mutual_info_score(y_true, y_pred),
            "homogeneity": metrics.homogeneity_score(y_true, y_pred)
        }

    for key, value in metrics_dict.items():
        print(f"{key.capitalize()}: {value:.4f}")


# =====================================================================
# Main Execution Entry Point
# =====================================================================
if __name__ == "__main__":
    # ---------------------------------------------------------
    # Step 1: Load parameters and initialize environment
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Dual-mode SOM Pipeline")
    parser.add_argument('--config', type=str, default='params.json', help="Path to JSON config")
    args = parser.parse_args()

    create_default_params(args.config)
    parameters = read_parameters(args.config)

    # Validate essential configuration keys
    required_keys = [
        'dataset_name', 'run_mode', 'train_data_path', 'test_data_path',
        'activation_distance', 'n_clusters', 'auto_find_clusters', 'k_min', 'k_max'
    ]
    missing_keys = [key for key in required_keys if key not in parameters]
    if missing_keys:
        print(f"\n[ERROR] Missing required parameters in JSON: {missing_keys}")
        sys.exit(1)

    dataset_name = parameters['dataset_name']
    run_mode = parameters['run_mode']

    # ---------------------------------------------------------
    # Step 2: Read and encode data (Dimensionality Reduction)
    # ---------------------------------------------------------
    train_data = get_dataset(parameters['train_data_path']) # p raw samples
    set_ae_args(parameters)                                 # Initialize encoder settings
    coded_data = encode_decode(train_data)                  # Returns tuple (X_latent, y_true)

    # ---------------------------------------------------------
    # Step 3: Create and train DualSOM model (Weight adjustment)
    # ---------------------------------------------------------
    model = DualSOM(parameters, coded_data)
    model.fit(coded_data)

    # Optional Unsupervised Feature: Dynamic Cluster Selection Trigger
    if run_mode == 'unsupervised' and parameters['auto_find_clusters']:
        selector = SOMClusterSelector(som_weights=model.som._weights)
        optimal_k = selector.evaluate(k_min=parameters['k_min'], k_max=parameters['k_max'])

        # Override structural parameters dynamically
        parameters['n_clusters'] = optimal_k
        model.parameters['n_clusters'] = optimal_k

    X_train, y_train = coded_data

    # ---------------------------------------------------------
    # Training Stage Predictions (Step 4a/4b)
    # ---------------------------------------------------------
    if run_mode == 'unsupervised':
        # Step 4a: Clustering (unsupervised boundary discovery)
        print("\n>>> Executing Stage 4a: Clustering Training Phase...")
        clusters = model.predict(coded_data, mode='clustering')
        evaluate_and_print(y_train, clusters, run_mode, dataset_name, "TRAINING")
    else:
        # Step 4b: Classification (supervised, labels available)
        print("\n>>> Executing Stage 4b: Classification Training Phase...")
        pred_labels = model.predict(coded_data, mode='classification')
        evaluate_and_print(y_train, pred_labels, run_mode, dataset_name, "TRAINING")

    # =========================================================
    # Post-Training: Generalization Phase
    # The identical DualSOM model and SAE encoder are applied to
    # new (unseen) test data. Workflow diverges based on mode.
    # =========================================================

    # ---------------------------------------------------------
    # Step 1: Read and encode test data (Using fitted SAE)
    # ---------------------------------------------------------
    test_data = get_dataset(parameters['test_data_path']) # n new samples
    coded_test = encode_decode(test_data)                 # Encode using pre-trained weights

    X_test, y_test = coded_test

    # ---------------------------------------------------------
    # Testing Stage Predictions (Step 2a/2b)
    # ---------------------------------------------------------
    if run_mode == 'unsupervised':
        # Step 2a: Clustering (unsupervised generalization)
        print("\n>>> Executing Stage 2a: Clustering Testing Phase...")
        clusters_test = model.predict(coded_test, mode='clustering')
        evaluate_and_print(y_test, clusters_test, run_mode, dataset_name, "TESTING")
    else:
        # Step 2b: Classification (supervised general recognition)
        print("\n>>> Executing Stage 2b: Classification Testing Phase...")
        pred_labels_test = model.predict(coded_test, mode='classification')
        evaluate_and_print(y_test, pred_labels_test, run_mode, dataset_name, "TESTING")

    print("\n>>> All Done.")
