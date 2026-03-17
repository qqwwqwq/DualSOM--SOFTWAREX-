import argparse
import math
import os
import sys
import json  # Used for handling external configuration files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import DataLoader, TensorDataset

# --- Custom Module Imports ---
# DualSom: The core Self-Organizing Map implementation
# preprocessing: Data loading utilities
# sparse_autoencoder: Sparse AE architecture and training loop
try:
    from Daulmap import DualSom
    from preprocessing import get_dataset
    from sparse_autoencoder import SparseAutoencoder, fit
except ImportError as e:
    print(f"Error: Could not import custom modules ({e}).")
    sys.exit(1)


# ==========================================
# [Configuration] JSON Parameter Loading
# ==========================================
def read_parameters(json_path):
    """Loads hyper-parameters from a JSON file to override script defaults."""
    if not os.path.exists(json_path):
        print(f"Warning: Config file '{json_path}' not found. Using script defaults.")
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file ({json_path}): {e}")
        return {}


def get_args():
    """Defines command-line arguments and merges them with JSON config."""
    parser = argparse.ArgumentParser(description="Sparse AE + DualSOM Data Recognition Pipeline")

    # Global Settings
    parser.add_argument('--config', type=str, default='params.json', help="Path to .json config file")
    parser.add_argument('--dataset_name', type=str, default='generic', help="'wut', 'pku', or 'generic'")
    parser.add_argument('--data_format', type=str, default='npy', choices=['npy', 'csv'])
    parser.add_argument('--input_dim', type=int, default=128, help="Feature dimension (auto-detected)")

    # Path Parameters
    parser.add_argument('--train_path', type=str, default="./data/train_data.npy")
    parser.add_argument('--train_label_path', type=str, default="./data/train_labels.npy")
    parser.add_argument('--test_path', type=str, default="./data/test_data.npy")
    parser.add_argument('--test_label_path', type=str, default="./data/test_labels.npy")
    parser.add_argument('--output_dir', type=str, default="./results")
    parser.add_argument('--reduction_factor', type=int, default=1, help="Downsampling factor for debugging")

    # Sparse Autoencoder (AE) Hyper-parameters
    parser.add_argument('--ae_batch_size', type=int, default=32)
    parser.add_argument('--ae_epochs', type=int, default=150)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--force_train_ae', type=int, default=1, help="1: Re-train AE; 0: Load existing weights")

    # SOM Hyper-parameters
    parser.add_argument('--som_size_index', type=float, default=10, help="Coefficient for grid scaling")
    parser.add_argument('--som_epochs', type=int, default=50)
    parser.add_argument('--som_sigma', type=float, default=4, help="Neighborhood radius")
    parser.add_argument('--som_lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--som_enable_validation', type=int, default=1)

    # Execution Mode
    parser.add_argument('--run_mode', type=str, default='supervised', choices=['supervised', 'unsupervised'])
    parser.add_argument('--n_clusters', type=int, default=5, help="Target clusters for unsupervised mode")

    # Merge Argument logic: Command line < JSON Config
    args, remaining_argv = parser.parse_known_args()
    if args.config:
        params = read_parameters(args.config)
        if params:
            print(f">>> Applying parameters from JSON: {args.config}")
            parser.set_defaults(**params)

    return parser.parse_args()


# ==========================================
# [Clustering] Secondary Grouping for SOM
# ==========================================
class SOMClusterer:
    """
    Used in Unsupervised Mode. Clusters the high-dimensional weight vectors 
    of SOM neurons into K discrete clusters using angular distance.
    """
    def __init__(self, n_clusters, max_iter=100, threshold=1e-4):
        self.K = n_clusters
        self.max_iter = max_iter
        self.threshold = threshold
        self.centroids = None
        self.labels_map = None

    def _angular_distance_matrix(self, W1, W2):
        """Calculates the angular distance (cosine-based) between two sets of vectors."""
        W1_norm = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-8)
        W2_norm = W2 / (np.linalg.norm(W2, axis=1, keepdims=True) + 1e-8)
        sim = np.clip(np.dot(W1_norm, W2_norm.T), -1.0, 1.0)
        ang = np.arccos(sim) * 180 / np.pi
        ang = np.minimum(ang, 180 - ang)
        return ang / 180 * np.pi

    def fit(self, som_weights):
        """Perform K-Means style clustering on SOM grid weights."""
        grid_x, grid_y, features = som_weights.shape
        U = grid_x * grid_y
        vw = som_weights.reshape(U, features)
        
        # Initial Centroid Selection (K-Means++ logic)
        dis = self._angular_distance_matrix(vw, vw)
        np.fill_diagonal(dis, 0)
        c1 = np.argmin(np.sum(dis, axis=0))
        centroids_idx = [c1]
        delta = dis[c1, :].copy()

        for i in range(1, self.K):
            last_c = centroids_idx[-1]
            delta = np.minimum(delta, dis[last_c, :])
            next_c = np.argmax(delta)
            centroids_idx.append(next_c)

        self.centroids = vw[centroids_idx].copy()
        
        # Iterative Optimization
        for it in range(self.max_iter):
            dist_to_centroids = self._angular_distance_matrix(vw, self.centroids)
            labels = np.argmin(dist_to_centroids, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.K):
                cluster_points = vw[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[k] = self.centroids[k]

            diff = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids
            if diff < self.threshold:
                print(f"      [Algorithm 2] Converged at iteration {it + 1}")
                break

        self.labels_map = labels.reshape(grid_x, grid_y)
        return self

    def predict(self, som, data_X):
        """Maps input data to the winning SOM neuron, then returns that neuron's cluster ID."""
        preds = []
        for x in data_X:
            win_x, win_y = som.winner(x)
            cluster_id = self.labels_map[win_x, win_y]
            preds.append(cluster_id)
        return np.array(preds)


# ==========================================
# [Helper] Prediction & Data Processing
# ==========================================
def classify_som(som, data, winmap):
    """Predicts labels by finding the most frequent class associated with a winning neuron."""
    from collections import Counter
    if not winmap:
        return [0] * len(data)
    
    # Calculate global majority class as a fallback
    default_class = sum(winmap.values(), Counter()).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


def load_and_process_data(args):
    """Loads raw files, applies MinMax normalization, and prepares Torch tensors."""
    print(f">>> Loading Data ({args.dataset_name})...")
    dataset_name = args.dataset_name.lower()
    data_dir = f"{dataset_name.upper()}_DATA"
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name in ['pku', 'wut']:
        # Legacy skeleton data processing
        train_processed_path = os.path.join(data_dir, "train_data.csv")
        test_processed_path = os.path.join(data_dir, "test_data.csv")
        trainData, trainlabel = get_dataset(raw_path=args.train_path, processed_path=train_processed_path,
                                            dataset_name=dataset_name)
        testData, testlabel = get_dataset(raw_path=args.test_path, processed_path=test_processed_path,
                                          dataset_name=dataset_name)

        X_train_raw = trainData.clone().detach().cpu().numpy()
        X_test_raw = testData.clone().detach().cpu().numpy()
        y_train = trainlabel.cpu().numpy()
        y_test = testlabel.cpu().numpy()
    else:
        # Standard vectorized matrix loading
        if args.data_format == 'npy':
            X_train_raw = np.load(args.train_path)
            y_train = np.load(args.train_label_path)
            X_test_raw = np.load(args.test_path)
            y_test = np.load(args.test_label_path)
        elif args.data_format == 'csv':
            X_train_raw = pd.read_csv(args.train_path).values
            y_train = pd.read_csv(args.train_label_path).values.squeeze()
            X_test_raw = pd.read_csv(args.test_path).values
            y_test = pd.read_csv(args.test_label_path).values.squeeze()

        # Flatten multi-dim data (e.g., from [N, H, W] to [N, H*W])
        if len(X_train_raw.shape) > 2:
            X_train_raw = X_train_raw.reshape(X_train_raw.shape[0], -1)
            X_test_raw = X_test_raw.reshape(X_test_raw.shape[0], -1)

    args.input_dim = X_train_raw.shape[1]

    # Apply data reduction for faster prototyping
    red = args.reduction_factor
    n_train, n_test = int(X_train_raw.shape[0]/red), int(X_test_raw.shape[0]/red)
    X_train_raw, y_train = X_train_raw[:n_train], y_train[:n_train]
    X_test_raw, y_test = X_test_raw[:n_test], y_test[:n_test]

    # Normalize inputs to [0, 1] for AE training
    input_scaler = MinMaxScaler()
    X_train_scaled = input_scaler.fit_transform(X_train_raw)
    X_test_scaled = input_scaler.transform(X_test_raw)

    return (torch.from_numpy(X_train_scaled).float(), y_train, 
            torch.from_numpy(X_test_scaled).float(), y_test)


def run_autoencoder(X_train, X_test, args):
    """Trains Sparse AE to compress data into sparse latent features."""
    print(f">>> Processing Sparse Autoencoder on {args.device}...")
    save_path = os.path.join('weight', f'{args.dataset_name}_AE_batch_{args.ae_batch_size}.pth')
    os.makedirs('weight', exist_ok=True)

    model = SparseAutoencoder(input_dim=args.input_dim).to(args.device)
    should_train = (args.force_train_ae == 1) or not os.path.exists(save_path)

    if should_train:
        model = fit(model, args.ae_batch_size, X_train, args.ae_epochs, args.device, save_path)
    else:
        model.load_state_dict(torch.load(save_path, map_location=args.device))

    def extract_features_batched(data_tensor):
        """Forward pass through Encoder to get latent features."""
        model.eval()
        loader = DataLoader(TensorDataset(data_tensor), batch_size=256, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(args.device)
                feat = model.fd(x) # fd: feature detection / encoder
                features_list.append(feat.cpu().numpy())
        return np.vstack(features_list)

    X_train_ae = extract_features_batched(X_train)
    X_test_ae = extract_features_batched(X_test)

    # Standardize features for SOM (Zero mean, unit variance)
    feature_scaler = StandardScaler()
    X_train_ae = feature_scaler.fit_transform(X_train_ae)
    X_test_ae = feature_scaler.transform(X_test_ae)

    return X_train_ae, X_test_ae, model


def run_som_training_and_eval(X_train, y_train, X_test, y_test, args):
    """Trains Dual-SOM and evaluates based on the selected run_mode."""
    N, M = X_train.shape[0], X_train.shape[1]

    # Heuristic calculation for grid size: sqrt(C * sqrt(N))
    size = math.ceil(np.sqrt(args.som_size_index * np.sqrt(N)))
    print(f">>> SOM Config: Grid Size={size}x{size}, Max Iter={args.som_epochs * N}")

    som = DualSom(size, size, M, sigma=args.som_sigma, learning_rate=args.som_lr,
                  repulsion=2, neighborhood_function='bubble',
                  activation_distance='angular')

    # Weight initialization using Principal Component Analysis
    som.pca_weights_init(X_train)
    max_iter = args.som_epochs * N
    
    # Train SOM
    som.train_batch(X_train, y_train, X_test, y_test, max_iter, 
                    verbose=True, enable_validation=(args.som_enable_validation == 1))

    # Evaluation Phase
    metrics_dict = {}
    if args.run_mode == 'supervised':
        print("\n=== Branch: Supervised Classification ===")
        winmap = som.labels_map(X_train, y_train)
        y_pred = classify_som(som, X_test, winmap)
        print(classification_report(y_test, np.array(y_pred)))
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "nmi": metrics.normalized_mutual_info_score(y_test, y_pred)
        }
    elif args.run_mode == 'unsupervised':
        print(f"\n=== Branch: Unsupervised Clustering (K={args.n_clusters}) ===")
        clusterer = SOMClusterer(n_clusters=args.n_clusters)
        clusterer.fit(som._weights)
        y_pred = clusterer.predict(som, X_test)
        metrics_dict = {
            "nmi": metrics.normalized_mutual_info_score(y_test, y_pred),
            "homogeneity": metrics.homogeneity_score(y_test, y_pred)
        }

    # Logging and Plotting
    np.save(os.path.join(args.output_dir, f"{args.dataset_name}_pred_{args.run_mode}.npy"), y_pred)
    visualize_results(X_test, y_test, y_pred, som, args)
    return som, metrics_dict


def visualize_results(X_test, y_test, y_pred, som, args):
    """Generates t-SNE scatter plots and SOM accuracy curves."""
    print(">>> Generating Visualizations...")
    limit = 1000
    idx = np.random.choice(len(X_test), min(len(X_test), limit), replace=False)
    H, C_pred, gd = X_test[idx], np.array(y_pred)[idx], np.array(y_test)[idx]

    # Dimensionality reduction for visualization
    tsne = TSNE(n_components=2, init='random', learning_rate=200.0, random_state=42)
    H_2D = tsne.fit_transform(H)

    plt.figure(figsize=(18, 5))

    # Plot 1: Prediction Clusters
    plt.subplot(1, 3, 1)
    plt.scatter(H_2D[:, 0], H_2D[:, 1], s=10, c=C_pred, cmap='tab10', alpha=0.7)
    plt.title(f"Prediction ({args.run_mode.capitalize()})")

    # Plot 2: Ground Truth
    plt.subplot(1, 3, 2)
    plt.scatter(H_2D[:, 0], H_2D[:, 1], s=10, c=gd, cmap='tab10', alpha=0.7)
    plt.title(f"Ground Truth ({args.dataset_name.upper()})")

    # Plot 3: SOM Validation Accuracy over Time
    plt.subplot(1, 3, 3)
    if hasattr(som, 'accuracy') and len(som.accuracy) > 0:
        plt.plot(som.it, som.accuracy)
        plt.title("SOM Training Accuracy")
    else:
        plt.text(0.5, 0.5, "Fast Mode\n(No Accuracy Curve)", ha='center', va='center')

    plt.savefig(os.path.join(args.output_dir, f'viz_{args.dataset_name}_{args.run_mode}.png'))
    plt.close()


if __name__ == "__main__":
    args = get_args()
    # Phase 1: Data Preparation
    X_train_raw, y_train, X_test_raw, y_test = load_and_process_data(args)
    # Phase 2: Sparse Feature Extraction (Deep Learning)
    X_train_ae, X_test_ae, ae_model = run_autoencoder(X_train_raw, X_test_raw, args)
    # Phase 3: Topology-Preserving Classification/Clustering (Competitive Learning)
    run_som_training_and_eval(X_train_ae, y_train, X_test_ae, y_test, args)
    print(">>> Pipeline Completed.")
