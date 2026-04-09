"""
Dualmap_api.py

Unified API for Sparse Autoencoder and Dual-mode Self-Organizing Map (Dual-SOM).

This module provides a complete, object-oriented pipeline for Latent Representation 
Learning (via PyTorch-based Sparse Autoencoder) and clustering/classification 
(via NumPy-based DualSOM). It is completely decoupled from CLI arguments and 
global states, making it suitable for standard Python imports, while maintaining 
strict method signature compatibility with legacy code.
"""

import os
import math
import numpy as np
from numpy import linalg, outer, meshgrid, einsum
from collections import defaultdict, Counter

# --- PyTorch Imports (Autoencoder) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Scikit-Learn Imports ---
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- Utilities ---
from tqdm import tqdm


# ==============================================================================
# PART 1: Sparse Autoencoder API
# ==============================================================================

class SparseAutoencoderModel(nn.Module):
    """
    Feed-forward Neural Network Architecture for the Sparse Autoencoder.
    Employs Batch Normalization, CELU (Continuously Differentiable Exponential 
    Linear Units) activations for smoother gradients, and a symmetric 
    encoder-decoder structure to force data compression.
    """
    def __init__(self, input_dim=57):
        """
        Initializes the network layers.
        
        Args:
            input_dim (int): Dimensionality of the raw input feature space.
        """
        super(SparseAutoencoderModel, self).__init__()

        # --- Encoder Section ---
        # Maps input features to an intermediate 72-dimensional space, then 
        # compresses them into a 36-dimensional Latent Space bottleneck.
        self.enc1 = nn.Linear(input_dim, 72)
        self.bn1 = nn.BatchNorm1d(72) # Stabilizes training by normalizing intermediate outputs
        self.enc2 = nn.Linear(72, 36)

        # --- Decoder Section ---
        # Attempts to reconstruct the original input from the 36D Latent Space.
        self.dec1 = nn.Linear(36, 72)
        self.bn2 = nn.BatchNorm1d(72)
        self.dec2 = nn.Linear(72, input_dim)

    def forward(self, x):
        """
        Standard forward pass used during training.
        Generates both the reconstructed inputs and the latent codes.

        Returns:
            tuple: (Reconstructed inputs scaled by sigmoid to [0,1], Latent vector).
        """
        x = F.celu(self.bn1(self.enc1(x)))
        latent = F.relu(self.enc2(x)) # ReLU ensures latent codes are non-negative
        x = F.celu(self.bn2(self.dec1(latent)))
        return torch.sigmoid(self.dec2(x)), latent

    def fd(self, x):
        """
        Feature extraction method used during the inference/test stage.
        Bypasses the decoder entirely to directly yield the latent code representation.
        """
        x = F.celu(self.bn1(self.enc1(x)))
        return F.relu(self.enc2(x))


class SparseAutoencoderAPI:
    """
    Python API wrapper for the Sparse Autoencoder.
    Maintains its own state (scalers, model weights) locally to avoid global variables.
    Includes legacy signatures (`set_ae_args`, `encode_decode`) for backward compatibility.
    """
    
    def __init__(self, parameters=None, **kwargs):
        """
        Initializes the autoencoder pipeline and preprocessing scalers.
        """
        self.model = None
        self.input_scaler = MinMaxScaler()    # Scales raw inputs to [0, 1] for neural network
        self.feature_scaler = StandardScaler() # Standardizes latent features (mean=0, variance=1) for SOM
        
        # Support legacy dictionary initialization if 'parameters' is provided
        if parameters is not None:
            self.set_ae_args(parameters)
        else:
            # Modern kwargs-based initialization with sensible defaults
            self.device = kwargs.get('device', 'cpu')
            self.epochs = kwargs.get('epochs', 50)
            self.batch_size = kwargs.get('batch_size', 64)
            self.learning_rate = kwargs.get('learning_rate', 1e-3)
            self.reg_param = kwargs.get('reg_param', 1e-4) # Penalty for L1 sparsity loss
            self.load_model = kwargs.get('load_model', False)
            self.model_path = kwargs.get('model_path', 'ae_model.pth')

    def set_ae_args(self, parameters):
        """LEGACY COMPATIBILITY: Updates configuration based on the old JSON params dictionary."""
        self.device = parameters.get('device', 'cpu')
        self.epochs = parameters.get('ae_epochs', 50)
        self.batch_size = parameters.get('ae_batch_size', 64)
        self.learning_rate = parameters.get('ae_lr', 1e-3)
        self.reg_param = parameters.get('ae_reg_param', 1e-4)
        self.load_model = parameters.get('ae_load_model', False)
        self.model_path = parameters.get('ae_model_path', 'ae_model.pth')

    def encode_decode(self, data):
        """
        LEGACY COMPATIBILITY: Matches the exact signature of the old global function.
        Dynamically routes to fit_transform() or transform() based on the internal model state.
        
        Args:
            data (tuple): A tuple containing (X_raw_features, y_labels).
        Returns:
            tuple: (X_latent_encoded, y_labels).
        """
        X_raw, y = data
        is_train = self.model is None # If model isn't built yet, we are in training mode
        
        if is_train:
            X_encoded = self.fit_transform(X_raw)
        else:
            X_encoded = self.transform(X_raw)
            
        return X_encoded, y

    def fit_transform(self, X):
        """
        Fits the data scalers, trains the autoencoder (or loads weights), 
        and returns the encoded latent features.
        """
        # Step 1: Scale raw inputs appropriately prior to network ingestion
        X_scaled = self.input_scaler.fit_transform(X)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        
        # Step 2: Initialize Model Architecture dynamically based on input shape
        input_dim = X.shape[1]
        self.model = SparseAutoencoderModel(input_dim=input_dim).to(self.device)
        
        # Step 3: Load existing weights or Train from scratch
        if self.load_model:
            if os.path.exists(self.model_path):
                print(f">>> Loading pre-trained AutoEncoder model from {self.model_path}...")
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            else:
                raise FileNotFoundError(f"Requested load_model=True, but no model found at {self.model_path}!")
        else:
            print(f">>> Training AutoEncoder from scratch for {self.epochs} epochs...")
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss() # Primary loss metric: Mean Squared Error for reconstruction

            # Construct PyTorch DataLoaders for reliable Mini-Batch training
            dataset = TensorDataset(X_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.model.train()
            for epoch in tqdm(range(self.epochs), desc="AE Training", unit="epoch"):
                for batch in loader:
                    x_batch = batch[0]
                    optimizer.zero_grad() # Clear previous gradients
                    outputs, latent = self.model(x_batch)
                    
                    # Total Loss = MSE (Reconstruction) + L1 Penalty (Forces Latent Sparsity)
                    loss = criterion(outputs, x_batch) + self.reg_param * torch.mean(torch.abs(latent))
                    loss.backward()
                    optimizer.step()

            # Persist model weights locally for future rapid loading
            os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            print(f">>> Model saved successfully to {self.model_path}")
            
        # Step 4: Extract and return Latent Features
        return self._extract_features(X_tensor, is_train=True)

    def transform(self, X):
        """
        Uses the pre-trained autoencoder and scalers to project new test data
        into the latent space.
        """
        if self.model is None:
            raise RuntimeError("Autoencoder is not trained. Call fit_transform() first.")
        
        # Transform using previously fitted scalers to prevent data leakage
        X_scaled = self.input_scaler.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        return self._extract_features(X_tensor, is_train=False)

    def _extract_features(self, X_tensor, is_train):
        """
        Internal helper to process tensor batches and standardize latent outputs.
        """
        self.model.eval() # Set model to evaluation mode (disables dropout/batchnorm updates)
        latent_list = []
        
        with torch.no_grad(): # Disable gradient calculation for faster inference
            dataset = TensorDataset(X_tensor)
            # Employ a larger, fixed batch size to optimize VRAM usage during mass inference
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            for batch in loader:
                x_batch = batch[0]
                latent_list.append(self.model.fd(x_batch).cpu().numpy())

        # Stack all batches into a single numpy array
        X_latent = np.vstack(latent_list)

        # Standardize the extracted latent codes prior to routing them into the SOM
        if is_train:
            return self.feature_scaler.fit_transform(X_latent)
        else:
            return self.feature_scaler.transform(X_latent)


# ==============================================================================
# PART 2: DualSOM Reviewer-Requested API Wrapper
# ==============================================================================

class DualSOM_API:
    """
    Python API wrapper designed to strictly fulfill legacy structural expectations.
    Allows passing original dictionaries and `coded_data` (X, y) tuples directly,
    while internally using the modern kwargs-based DualSOM engine.
    """
    
    def __init__(self, parameters, coded_data=None):
        """LEGACY COMPATIBILITY: Matches original `__init__(parameters, coded_data)` signature."""
        self.params = parameters
        self.model = None

    def fit(self, coded_data):
        """LEGACY COMPATIBILITY: Takes `coded_data` tuple and initiates training."""
        X, y = coded_data
        # Unpack the dictionary directly into kwargs
        self.model = DualSOM(**self.params)
        self.model.fit(X, y=y)

    def predict(self, coded_data, mode='clustering'):
        """LEGACY COMPATIBILITY: Takes `coded_data` tuple and returns predictions."""
        X, _ = coded_data
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        return self.model.predict(X, mode=mode)


# ==============================================================================
# PART 3: DualSOM Core High-Level Engine
# ==============================================================================

def thumb_rule(data_len, som_size_index):
    """
    Calculates the optimal topological grid size for the SOM based on dataset volume.
    Formula: ceil(sqrt(index * sqrt(N)))
    """
    return math.ceil(np.sqrt(som_size_index * np.sqrt(data_len)))

class DualSOM:
    """
    Unified high-level engine for the Dual-mode SOM.
    Manages the lifecycle of the base map grid, weight initialization, and clustering algorithms.
    """

    def __init__(self, **kwargs):
        """
        Initializes the DualSOM engine with modern keyword arguments.
        Intelligently resolves parameter names to support both old and new conventions.
        """
        self.clusterer = None
        self._winmap = None
        self._train_labels = None

        # Parameter mapping dictionary accommodating both modern API calls and legacy JSON keys
        self.parameters = {
            'run_mode': kwargs.get('run_mode', 'clustering'),
            'som_load_model': kwargs.get('som_load_model', kwargs.get('load_model', False)),
            'som_model_path': kwargs.get('som_model_path', kwargs.get('model_path', 'som_model.npy')),
            'som_size_index': kwargs.get('som_size_index', 2.0),
            'som_sigma': kwargs.get('som_sigma', kwargs.get('sigma', None)),
            'som_sigma_target': kwargs.get('som_sigma_target', kwargs.get('sigma_target', 0.1)),
            'som_lr': kwargs.get('som_lr', kwargs.get('learning_rate', 0.5)),
            'som_lr_target': kwargs.get('som_lr_target', kwargs.get('lr_target', 0.05)),
            'activation_distance': kwargs.get('activation_distance', 'euclidean'),
            'som_epochs': kwargs.get('som_epochs', kwargs.get('epochs', 100)),
            'som_enable_validation': kwargs.get('som_enable_validation', kwargs.get('enable_validation', False)),
            'n_clusters': kwargs.get('n_clusters', 2),
            'kmeans_max_iter': kwargs.get('kmeans_max_iter', 300),
            'kmeans_threshold': kwargs.get('kmeans_threshold', 1e-4)
        }
        
        self.run_mode = self.parameters['run_mode']
        self.load_model = self.parameters['som_load_model']
        self.model_path = self.parameters['som_model_path']
        self.grid_size = kwargs.get('grid_size', None)
        self.som = None 

    def fit(self, X, y=None):
        """
        Trains the SOM grid weights using the provided feature matrix X.
        Delays the actual grid creation until X is provided so dimensions are known.
        """
        self._train_labels = y

        if self.som is None:
            # Auto-calculate grid size using thumb rule if not provided
            if self.grid_size is None:
                self.grid_size = thumb_rule(len(X), self.parameters['som_size_index'])
            
            # Heuristic default for neighborhood radius (sigma): usually 1/4 of the map size
            if self.parameters['som_sigma'] is None:
                self.parameters['som_sigma'] = max(1.0, self.grid_size / 4.0)

            print(f"   [SOM Init] Grid Size: {self.grid_size}x{self.grid_size} | Features: {X.shape[1]}")
            self.som = BaseDualSom(
                self.grid_size, self.grid_size, input_len=X.shape[1],
                sigma=self.parameters['som_sigma'],
                sigma_target=self.parameters['som_sigma_target'],
                learning_rate=self.parameters['som_lr'],
                lr_target=self.parameters['som_lr_target'],
                activation_distance=self.parameters['activation_distance']
            )

        if self.load_model:
            if os.path.exists(self.model_path):
                print(f"\n>>> Loading pre-trained SOM weights from {self.model_path}...")
                self.som._weights = np.load(self.model_path)
            else:
                raise FileNotFoundError(f"Requested load_model: True, but no model found at {self.model_path}")
        else:
            print("\n>>> Training SOM from scratch...")
            # Initialize weights utilizing PCA to dramatically accelerate topological convergence
            self.som.pca_weights_init(X)

            max_iter = self.parameters['som_epochs'] * len(X)
            enable_val = bool(self.parameters['som_enable_validation'])
            
            # Short-circuit validation to save time if operating in unsupervised clustering mode
            if self.run_mode == 'unsupervised':
                enable_val = False

            self.som.train_batch(X, y, X, y, max_iter, verbose=True, enable_validation=enable_val)

            # Serialize and save weights locally
            os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)
            np.save(self.model_path, self.som._weights)
            print(f">>> SOM weights saved successfully to {self.model_path}")

    def predict(self, X, mode=None):
        """
        Maps data to the SOM grid and assigns final predictions.
        Handles both Clustering (via K-Means on weights) and Classification (via Majority Voting).
        """
        eval_mode = mode if mode is not None else self.run_mode
        
        if eval_mode == 'clustering':
            # Initialize and fit the weight-space K-Means clusterer if not already trained
            if self.clusterer is None:
                self.clusterer = SOMClusterer(
                    n_clusters=self.parameters['n_clusters'], 
                    max_iter=self.parameters['kmeans_max_iter'], 
                    threshold=self.parameters['kmeans_threshold']
                )
                self.clusterer.fit(self.som._weights)
                
            # Prediction routes data to Best Matching Unit (BMU) -> assigns BMU's cluster ID to data
            return self.clusterer.predict(self.som, X)

        elif eval_mode == 'classification':
            if self._train_labels is None:
                 raise ValueError("Cannot run 'classification' prediction without target labels 'y'.")
                 
            # Construct the BMU label voting map if not already built
            if self._winmap is None:
                self._winmap = self.som.labels_map(X, self._train_labels)

            # Establish a global fallback class for edge-cases where a test sample hits a virgin/empty neuron
            default_class = sum(self._winmap.values(), Counter()).most_common()[0][0]
            result = []
            
            for d in X:
                win_pos = self.som.winner(d)
                if win_pos in self._winmap:
                    result.append(self._winmap[win_pos].most_common()[0][0]) # Majority vote
                else:
                    result.append(default_class)
            return np.array(result)
            
    def get_weights(self):
        """Returns the trained SOM weight matrix (Useful for visualizations or API interactions)."""
        if self.som is not None:
            return self.som._weights
        return None


# ==============================================================================
# PART 4: DualSOM Low-Level Core Mathematical Classes
# ==============================================================================

class SOMClusterer:
    """
    A modified K-Means clustering algorithm adapted specifically for grouping
    Self-Organizing Map neuron weight vectors. Utilizes angular distance matrices.
    """
    def __init__(self, n_clusters, max_iter, threshold):
        self.K = n_clusters
        self.max_iter = max_iter
        self.threshold = threshold # Early stopping threshold based on centroid displacement
        self.labels_map = None
        self.centroids = None

    def _angular_distance_matrix(self, W1, W2):
        """
        Computes the pairwise angular distance matrix between two sets of vectors.
        Angular distance provides better grouping for high-dimensional feature spaces.
        """
        # Normalize vectors to unit length to compute cosine similarity safely
        W1_norm = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-8)
        W2_norm = W2 / (np.linalg.norm(W2, axis=1, keepdims=True) + 1e-8)
        
        sim = np.clip(np.dot(W1_norm, W2_norm.T), -1.0, 1.0)
        ang = np.arccos(sim) * 180 / np.pi # Convert to degrees
        ang = np.minimum(ang, 180 - ang)   # Handle reflections
        return ang / 180 * np.pi           # Convert back to radians for distance metric

    def fit(self, som_weights):
        """
        Executes the K-Means algorithm over the converged SOM weight grid.
        Flattens the grid, initializes centroids using Farthest-First Traversal,
        and clusters the map nodes.
        """
        grid_x, grid_y, features = som_weights.shape
        U = grid_x * grid_y
        vw = som_weights.reshape(U, features)
        
        dis = self._angular_distance_matrix(vw, vw)
        np.fill_diagonal(dis, 0)

        # Farthest-first traversal strategy (K-Means++ style) for robust initial centroid selection
        c1 = np.argmin(np.sum(dis, axis=0))
        centroids_idx = [c1]
        delta = dis[c1, :].copy()

        for _ in range(1, self.K):
            last_c = centroids_idx[-1]
            delta = np.minimum(delta, dis[last_c, :])
            next_c = np.argmax(delta)
            centroids_idx.append(next_c)

        self.centroids = vw[centroids_idx].copy()
        labels = np.zeros(U, dtype=int)

        # Main assignment and update loop
        for _ in range(self.max_iter):
            dist_to_centroids = self._angular_distance_matrix(vw, self.centroids)
            labels = np.argmin(dist_to_centroids, axis=1) # Assign points to closest centroid

            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.K):
                cluster_points = vw[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0) # Update centroid to mean
                else:
                    new_centroids[k] = self.centroids[k] # Handle empty clusters

            # Check for convergence
            diff = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids
            if diff < self.threshold:
                break

        # Reshape flat labels back to 2D grid coordinates
        self.labels_map = labels.reshape(grid_x, grid_y)
        return self

    def predict(self, som, data_X):
        """
        Projects raw data onto the SOM and retrieves the cluster ID of the winning neuron.
        """
        preds = []
        for x in data_X:
            win_x, win_y = som.winner(x)
            preds.append(self.labels_map[win_x, win_y])
        return np.array(preds)


class BaseDualSom(object):
    """
    Low-level mathematical implementation of the Self-Organizing Map grid.
    Augmented with dynamic distance routing (Euclidean, Cosine, Angular) and
    an exponential attention mechanism for robust weight updates.
    """
    def __init__(self, x, y, input_len, sigma, sigma_target,
                 learning_rate, lr_target, activation_distance):
        self._learning_rate = learning_rate
        self._lr_target = lr_target
        self._sigma = sigma
        self._sigma_target = sigma_target
        self._input_len = input_len

        # Initialization of map weights randomly on a unit hypersphere
        self._weights = np.random.rand(x, y, input_len) * 2 - 1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = np.zeros((x, y))
        
        # Create coordinate grids to calculate topological distances efficiently
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)
        self._xx, self._yy = meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)

        # Dictionary routing for dynamic distance metric selection
        distance_functions = {
            "angular": self._angular_distance,
            "euclidean": self._euclidean_distance,
            "cosine": self._cosine_distance
        }

        if activation_distance not in distance_functions:
            msg = f"'{activation_distance}' not supported. Distances available: {', '.join(distance_functions.keys())}"
            raise ValueError(msg)

        self._activation_distance = distance_functions[activation_distance]

    def _euclidean_distance(self, x, w):
        """Standard L2 Euclidean distance metric."""
        return linalg.norm(np.subtract(x, w), axis=-1)

    def _cosine_distance(self, x, w):
        """Cosine distance metric (1 - Cosine Similarity). Ideal for sparse high-dimensional data."""
        num = (w * x).sum(axis=2)
        denum = np.multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum + 1e-8)

    def _angular_distance(self, x, w):
        """Angular distance metric derived from cosine similarity. Ideal for directional data."""
        num = (w * x).sum(axis=2)
        denum = np.multiply(linalg.norm(w, axis=2), linalg.norm(x))
        res = num / (denum + 1e-8)
        res = np.clip(res, -1.0, 1.0)
        i = np.arccos(res) * 180 / np.pi
        tt = 180 - i
        res = np.minimum(i, tt)
        return res / 180 * np.pi

    def _activate(self, x):
        """Computes the activation distance map across the entire grid for input vector x."""
        self._activation_map = self._activation_distance(x, self._weights)

    def winner(self, x):
        """Identifies the (x, y) grid coordinates of the Best Matching Unit (BMU)."""
        self._activate(x)
        return np.unravel_index(self._activation_map.argmin(), self._activation_map.shape)

    def update(self, x, win, t, max_iteration):
        """
        Updates the synaptic weights of the map.
        Incorporates an exponential attention mechanism to penalize node updates
        if their activation distance to the input is exceedingly large.
        """
        tt = t / max_iteration
        
        # Apply exponential decay to learning rate and neighborhood radius over time
        eta = self._learning_rate * (self._lr_target / self._learning_rate) ** tt
        sig = self._sigma * (self._sigma_target / self._sigma) ** tt

        # Compute the spatial neighborhood bounding box around the BMU
        ax = np.logical_and(self._neigx > win[0] - sig, self._neigx < win[0] + sig)
        ay = np.logical_and(self._neigy > win[1] - sig, self._neigy < win[1] + sig)
        g = outer(ax, ay) * 1.0 * eta

        # Apply exponential attention decay based on topological activation map
        gg = self._activation_map.copy()
        # Einstein summation (einsum) computes multilinear tensor operations efficiently
        G = einsum('ij, ij->ij', g, np.exp(-gg + np.pi))

        # Perform the vectorized competitive weight update (pulling nodes towards input x)
        self._weights += einsum('ij, ijk->ijk', G, x - self._weights)

    def pca_weights_init(self, data):
        """
        Initializes the SOM weights to span the first two principal components
        of the dataset. By aligning the initial grid with the data's highest variance,
        topological convergence is drastically accelerated.
        """
        # Compute eigenvalues and eigenvectors of the covariance matrix
        pc_length, pc = linalg.eig(np.cov(np.transpose(data)))
        pc_order = np.argsort(-pc_length) # Sort by variance
        
        # Distribute grid weights evenly along the 2 largest principal components
        for i, c1 in enumerate(np.linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(np.linspace(-1, 1, len(self._neigy))):
                self._weights[i, j] = c1 * pc[:, pc_order[0]] + c2 * pc[:, pc_order[1]]

    def train_batch(self, data, Y_train=None, X_test=None, Y_test=None, max_iter=1000, verbose=False,
                    enable_validation=True):
        """
        The main training loop executed over the generated sequence of iterations.
        """
        iterations = np.arange(max_iter) % len(data)

        idx = 1
        for t, iteration in enumerate(tqdm(iterations, desc="SOM Training", unit="iter")):
            # Periodic accuracy validation checks (only applicable to supervised mode)
            if enable_validation and t == idx * len(data) * 5:
                winmap = self.labels_map(data, Y_train)
                default_class = Counter(Y_train).most_common(1)[0][0]
                result = []
                for d in X_test:
                    win_pos = self.winner(d)
                    result.append(winmap[win_pos].most_common()[0][0] if win_pos in winmap else default_class)
                acc = round(accuracy_score(Y_test, result) * 100, 2)
                tqdm.write(f"Validation at step {idx}: Accuracy {acc}%")
                idx += 1

            # Update weights for the current sample
            self.update(data[iteration], self.winner(data[iteration]), t, max_iter)

    def labels_map(self, data, labels):
        """
        Constructs a voting map associating each grid neuron with the class labels
        of the training samples that map to it. Used for classification predictions.
        """
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.winner(x)].append(l) # Log the label into the winning neuron's list
        for pos in winmap:
            winmap[pos] = Counter(winmap[pos]) # Convert lists to frequency counters
        return winmap
