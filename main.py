"""
Main execution pipeline for the Dual-mode Self-Organizing Map (Dual-SOM).

This script serves as the primary entry point for the repository. It handles
JSON-based configuration loading, dataset ingestion (preprocessed data),
autoencoder-based dimensionality reduction, and executes the 5-stage workflow
described in the paper. It seamlessly supports both supervised (classification)
and unsupervised (clustering) operational modes.
"""

import argparse
import json
import os
import sys
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from preprocessing import get_dataset
from sparse_autoencoder import encode_decode, set_ae_args
from Dualmap import DualSOM

# ==========================================
# Comprehensive Parameters List (All-in-JSON)
# ==========================================
SUGGESTED_PARAMETERS = {
    # Workflow Selection
    "dataset_name": "wut",          # Target dataset identifier [Values: 'wut', 'pku', 'mnist', etc.]
    "run_mode": "supervised",       # Execution mode [Values: 'supervised' or 'unsupervised']
    "device": "cuda",               # Hardware acceleration [Values: 'cuda', 'cpu']

    # Data Paths
    "train_data_path": "Datas/WUT/train_data.csv",
    "test_data_path": "Datas/WUT/test_data.csv",

    # SOM Hyperparameters
    "som_size_index": 5.0,          # Grid size using the heuristic rule: S ≈ som_size_index * sqrt(P), where P is sample size. [Range: 1.0 - 10.0, Suggested: 5.0]
    "som_epochs": 50,               # SOM training passes [e.g. 50, 100, 200...]
    "som_sigma": 4.0,               # Initial neighborhood radius [Range: 1.0 - 10.0, Suggested: 4.0]
    "som_sigma_target": 0.01,       # Asymptotic target for radius decay [Range: 0.001 - 0.1, Suggested: 0.01]
    "som_lr": 0.1,                  # Initial learning rate [Range: 0.01 - 1.0, Suggested: 0.1 - 0.5]
    "som_lr_target": 0.001,         # Asymptotic target for LR decay [Range: 0.0001 - 0.01, Suggested: 0.001]
    "activation_distance": "angular", # BMU distance metric [Values: 'angular':  best for directional / skeletal data, 'euclidean': general-purpose, 'cosine': best for high-dimensional sparse dat]
    "som_enable_validation": 1,     # Enable/disable periodic validation prints [Values: 1 (True) or 0 (False)]
    "som_load_model": False,        # Switch to bypass training and load a pre-trained SOM [Values: True, False]
    "som_model_path": "weight/som_weights.npy",  # Filepath for saving/loading the SOM weight matrix

    # Clustering Hyperparameters (Active only in 'unsupervised' mode)
    "auto_find_clusters": False,    # [NEW SWITCH] Dynamically calculate optimal K [Values: True, False]
    "k_min": 2,                     # [NEW PARAM] Min K to evaluate if auto_find_clusters is True [Range: > 1, Suggested: 2]
    "k_max": 10,                    # [NEW PARAM] Max K to evaluate if auto_find_clusters is True [Range: > k_min]
    "n_clusters": 10,               # Custom target number of clusters [Suggested: Matches expected dataset classes, or auto-seleted by our method.]
    "kmeans_max_iter": 100,         # Maximum iterations for K-Means convergence [Range: 100 - 1000, Suggested: 100 - 300]
    "kmeans_threshold": 1e-4,       # Convergence threshold (centroid shift) for K-Means [Range: 1e-5 - 1e-2, Suggested: 1e-4]

    # Sparse Autoencoder (SAE) Hyperparameters
    "ae_batch_size": 32,            # Mini-batch size for SAE gradient descent [Values: 16, 32, 64, 128, 256. Suggested: 32 or 64]
    "ae_epochs": 150,               # Number of training epochs for the SAE [Range: 50 - 500, Suggested: 150]
    "ae_lr": 0.001,                 # Learning rate for the Adam optimizer [Range: 1e-4 - 1e-2, Suggested: 0.001]
    "ae_reg_param": 0.001,          # Coefficient for the L1 sparsity penalty [Range: 1e-5 - 1e-1, Suggested: 0.001]
    "ae_load_model": False,         # Switch to bypass training and load pre-trained SAE weights [Values: True, False]
    "ae_model_path": "weight/sparse_ae.pth",  # Filepath for saving/loading the SAE PyTorch model
}

# =====================================================================
# Integrated Cluster Selection Logic
# =====================================================================
class SOMClusterSelector:
    """
    Evaluates and selects the optimal cluster number (k) based on SOM weights
    and angular distance implementation.
    """
    def __init__(self, som_weights):
        self.som_weights = som_weights

    def _angular_distance_matrix(self, W1, W2):
        W1_norm = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-8)
        W2_norm = W2 / (np.linalg.norm(W2, axis=1, keepdims=True) + 1e-8)
        sim = np.clip(np.dot(W1_norm, W2_norm.T), -1.0, 1.0)
        ang = np.arccos(sim) * 180 / np.pi
        ang = np.minimum(ang, 180 - ang)
        return ang / 180 * np.pi

    def _compute_angular_L_score(self, weights, labels, centroids):
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
        if k_min <= 1:
            k_min = 2 

        weights = self.som_weights
        if len(weights.shape) > 2:
            weights = weights.reshape(-1, weights.shape[-1])

        normalized_weights = normalize(weights, norm='l2')
        k_range = list(range(k_min, k_max + 1))
        delta_L_scores = []

        print(f"\n>>> Auto-Evaluating optimal clusters from K={k_min} to {k_max}...")
        eval_range = list(range(k_min - 1, k_max + 1))
        temp_L_scores = {}

        for k in eval_range:
            kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=1, n_init=10)
            y_pred_weights = kmeans.fit_predict(normalized_weights)
            centroids = kmeans.cluster_centers_
            temp_L_scores[k] = self._compute_angular_L_score(weights, y_pred_weights, centroids)

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
    if not os.path.exists(json_path):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(SUGGESTED_PARAMETERS, f, indent=4)
        print(f"Created default parameter file at {json_path}")

def read_parameters(json_path):
    if not os.path.exists(json_path):
        print(f"Warning: Config '{json_path}' not found. Using script defaults.")
        return SUGGESTED_PARAMETERS.copy()
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_and_print(y_true, y_pred, mode, dataset, stage_label):
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

    for k, v in metrics_dict.items():
        print(f"{k.capitalize()}: {v:.4f}")

if __name__ == "__main__":
    # --- Step 1: Load parameters
    parser = argparse.ArgumentParser(description="Dual-mode SOM Pipeline with JSON config")
    parser.add_argument('--config', type=str, default='params.json', help="Path to the JSON configuration file")
    args = parser.parse_args()

    create_default_params(args.config)
    parameters = read_parameters(args.config)

    # ----------------------------------------------------------
    # Strict validation of required parameters
    # ----------------------------------------------------------
    required_keys = [
        'dataset_name', 'run_mode', 'train_data_path', 'test_data_path', 
        'activation_distance', 'n_clusters', 'auto_find_clusters', 'k_min', 'k_max'
    ]

    missing_keys = [key for key in required_keys if key not in parameters]
    
    if missing_keys:
        print("\n" + "!"*50)
        print("[ERROR] The following required parameters are missing from the JSON configuration:")
        for key in missing_keys:
            print(f"  - '{key}'")
        print("Please add these parameters to your config file and try again. Execution aborted.")
        print("!"*50 + "\n")
        sys.exit(1)

    # Strictly use dictionary indexing without fallback defaults
    dataset_name = parameters['dataset_name']
    run_mode = parameters['run_mode']
    train_data_path = parameters['train_data_path']
    test_data_path = parameters['test_data_path']
    activation_distance = parameters['activation_distance']
    auto_find_clusters = parameters['auto_find_clusters']
    n_clusters = parameters['n_clusters']

    print(f"\n--- Workflow Initialization ---")
    print(f"Dataset: {dataset_name.upper()} | Mode: {run_mode.upper()}")
    print(f"BMU Distance Metric: '{activation_distance}'")
    if run_mode == 'unsupervised':
        print(f"Cluster Selection: {'AUTO' if auto_find_clusters else 'CUSTOM'} (n_clusters = {n_clusters})")
    print(f"-------------------------------\n")

    # --- Step 2: Read and encode data ---
    train_data = get_dataset(train_data_path)
    set_ae_args(parameters)
    coded_data = encode_decode(train_data)

    # --- Step 3: Create and train DualSOM model (weight adjustment) ---
    model = DualSOM(parameters, coded_data)
    model.fit(coded_data)

    # ==========================================================
    # --- NEW: Dynamic Cluster Selection Trigger ---
    # ==========================================================
    if run_mode == 'unsupervised' and auto_find_clusters:
        selector = SOMClusterSelector(som_weights=model.som._weights)
        optimal_k = selector.evaluate(k_min=parameters['k_min'], k_max=parameters['k_max'])
        
        # Override the n_clusters dynamically in both main dict and model internals
        parameters['n_clusters'] = optimal_k
        model.parameters['n_clusters'] = optimal_k
        print(f">>> Proceeding with dynamically selected K={optimal_k}")

    # --- Step 4: Mode-specific implementation ---
    X_train, y_train = coded_data
    if run_mode == 'unsupervised':
        print("\n>>> Executing Stage 4a: Clustering Training Phase...")
        y_pred_train = model.predict(coded_data, mode='clustering')
    else:
        print("\n>>> Executing Stage 4b: Classification Training Phase...")
        y_pred_train = model.predict(coded_data, mode='classification')
        
    evaluate_and_print(y_train, y_pred_train, run_mode, dataset_name, "TRAINING")

    # --- Step 5: Read, encode, and predict on test data ---
    test_data = get_dataset(test_data_path)
    coded_test = encode_decode(test_data)
    X_test, y_test = coded_test
    
    if run_mode == 'unsupervised':
        print("\n>>> Executing Stage 5a: Clustering Testing Phase...")
        y_pred_test = model.predict(coded_test, mode='clustering')
    else:
        print("\n>>> Executing Stage 5b: Classification Testing Phase...")
        y_pred_test = model.predict(coded_test, mode='classification')

    # Output Testing Metrics
    evaluate_and_print(y_test, y_pred_test, run_mode, dataset_name, "TESTING")

    print("\n>>> All Done.")
