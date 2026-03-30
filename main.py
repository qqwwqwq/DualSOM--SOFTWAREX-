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
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

from preprocessing import get_dataset
from sparse_autoencoder import encode_decode, set_ae_args
from Dualmap import DualSOM

# ==========================================
# Comprehensive Parameters List (All-in-JSON)
# ==========================================
SUGGESTED_PARAMETERS = {
    # Workflow Selection
    "dataset_name": "wut",          # Target dataset identifier (e.g., 'wut', 'pku', 'mnist')
    "run_mode": "supervised",       # Execution mode: 'supervised' or 'unsupervised'
    "device": "cuda",               # Hardware acceleration: 'cuda' or 'cpu'

    # Data Paths
    "train_data_path": "Datas/WUT/train_data.csv",
    "test_data_path": "Datas/WUT/test_data.csv",

    # SOM Hyperparameters
    "som_size_index": 10.0,         # Multiplier for the grid size heuristic calculation
    "som_epochs": 50,               # Number of complete passes over the dataset during SOM training
    "som_sigma": 4.0,               # Initial neighborhood radius for weight updates
    "som_sigma_target": 0.01,       # Asymptotic target for the neighborhood radius decay
    "som_lr": 0.1,                  # Initial learning rate
    "som_lr_target": 0.001,         # Asymptotic target for the learning rate decay
    "activation_distance": "angular", # Distance metric for Best Matching Unit (BMU) selection
    "som_enable_validation": 1,     # Flag to enable/disable periodic validation prints (1 or 0)
    "som_load_model": False,                     # Switch to bypass training and load a pre-trained SOM
    "som_model_path": "weight/som_weights.npy",  # Filepath for saving/loading the SOM weight matrix

    # Clustering Hyperparameters (Active only in 'unsupervised' mode)
    "n_clusters": 10,               # Target number of clusters for the weight-space K-Means
    "kmeans_max_iter": 100,         # Maximum iterations for K-Means convergence
    "kmeans_threshold": 1e-4,       # Convergence threshold (centroid shift) for K-Means

    # Sparse Autoencoder (SAE) Hyperparameters
    "ae_batch_size": 32,            # Mini-batch size for SAE gradient descent
    "ae_epochs": 150,               # Number of training epochs for the SAE
    "ae_lr": 0.001,                 # Learning rate for the Adam optimizer
    "ae_reg_param": 0.001,          # Coefficient for the L1 sparsity penalty on the latent space
    "ae_load_model": False,                   # Switch to bypass training and load pre-trained SAE weights
    "ae_model_path": "weight/sparse_ae.pth",  # Filepath for saving/loading the SAE PyTorch model
    "reduction_factor": 1           # Factor to subset data for rapid debugging (1 = full dataset)
}

def create_default_params(json_path):
    """Generates a default JSON configuration file if one does not exist."""
    if not os.path.exists(json_path):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(SUGGESTED_PARAMETERS, f, indent=4)
        print(f"Created default parameter file at {json_path}")

def read_parameters(json_path):
    """Reads and parses hyperparameters from the specified JSON configuration file."""
    if not os.path.exists(json_path):
        print(f"Warning: Config '{json_path}' not found. Using script defaults.")
        return SUGGESTED_PARAMETERS.copy()
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_and_print(y_true, y_pred, mode, dataset, stage_label):
    """Computes and prints formatted evaluation metrics based on the execution mode."""
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
    parser = argparse.ArgumentParser(description="Dual-mode SOM Pipeline with JSON config")
    parser.add_argument('--config', type=str, default='params.json', help="Path to the JSON configuration file")
    args = parser.parse_args()

    # Initialization: Load configurations
    create_default_params(args.config)
    parameters = read_parameters(args.config)

    dataset_name = parameters.get('dataset_name', 'mnist')
    run_mode = parameters.get('run_mode', 'supervised')

    train_data_path = parameters.get('train_data_path', 'data/train_data.csv')
    test_data_path = parameters.get('test_data_path', 'data/test_data.csv')

    # Propagate parsed parameters to the Autoencoder module's global state
    set_ae_args(parameters)

    # ==========================================================
    # Example Workflow (Mapped exactly to the 5 paper stages)
    # ==========================================================

    # --- Stage 1 & 2: Data preprocessing & Latent representation learning ---
    train_data = get_dataset(train_data_path, is_train=True, dataset_name=dataset_name)
    coded_data = encode_decode(train_data)

    # --- Stage 3: SOM weight adjustment ---
    # The map initializes itself, automatically calculating the optimal grid size internally.
    model = DualSOM(parameters, coded_data)
    model.fit(coded_data)

    # --- Stage 4: Mode-specific training ---
    X_train, y_train = coded_data

    if run_mode == 'unsupervised':
        print("\n>>> Executing Stage 4a: Clustering Training Phase...")
        y_pred_train = model.predict(coded_data, mode='clustering')
    else:
        print("\n>>> Executing Stage 4b: Classification Training Phase...")
        y_pred_train = model.predict(coded_data, mode='classification')

    # --- Stage 5: Prediction on new data ---
    test_data = get_dataset(test_data_path, is_train=False, dataset_name=dataset_name)
    coded_test = encode_decode(test_data)
    X_test, y_test = coded_test

    if run_mode == 'unsupervised':
        print("\n>>> Executing Stage 5a: Clustering Testing Phase...")
        y_pred_test = model.predict(coded_test, mode='clustering')
    else:
        print("\n>>> Executing Stage 5b: Classification Testing Phase...")
        y_pred_test = model.predict(coded_test, mode='classification')

    # ==========================================================
    # Post-processing Metrics Output
    # ==========================================================

    # 1. Output Training Metrics (Evaluates representation capacity)
    evaluate_and_print(y_train, y_pred_train, run_mode, dataset_name, "TRAINING")

    # 2. Output Testing Metrics (Evaluates generalization capacity)
    evaluate_and_print(y_test, y_pred_test, run_mode, dataset_name, "TESTING")

    print("\n>>> All Done.")
