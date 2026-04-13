"""
Dual-mode Self-Organizing Map (Dual-SOM) Core Implementation.

This module provides the mathematical and computational foundation for the SOM.
It contains the low-level grid operations (`BaseDualSom`), a specialized K-Means
algorithm designed for grouping angular neuron weights (`SOMClusterer`), and a
unified high-level wrapper (`DualSOM`) that interfaces seamlessly with the main pipeline.
"""

import os
import math
import numpy as np
from numpy import linalg, outer, meshgrid, einsum
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def thumb_rule(data_len, som_size_index):
    """
    Calculates the optimal grid size for the SOM based on dataset volume.

    Uses a well-established heuristic where the total number of neurons (S^2)
    scales with the square root of the number of training samples.

    Args:
        data_len (int): Total number of samples in the training dataset (N).
        som_size_index (float): A scaling hyperparameter.

    Returns:
        int: The dimension (width and height) of the square SOM grid (S).
    """
    return math.ceil(np.sqrt(som_size_index * np.sqrt(data_len)))


class DualSOM:
    """
    Unified high-level wrapper for the Dual-mode SOM.

    Encapsulates the underlying base map (`BaseDualSom`) and the weight-space
    clusterer (`SOMClusterer`), exposing standard scikit-learn style `fit()`
    and `predict()` APIs to the main workflow.

    Attributes:
        parameters (dict): The hyperparameter dictionary loaded from JSON.
        run_mode (str): Operational mode ('supervised' or 'unsupervised').
        load_model (bool): Flag indicating if weights should be loaded from disk.
        model_path (str): Filepath for serializing/deserializing map weights.
        grid_size (int): The calculated width and height of the SOM grid.
        som (BaseDualSom): The underlying mathematical SOM instance.
        clusterer (SOMClusterer or None): The unsupervised weight-space clusterer.
    """

    def __init__(self, parameters, coded_data):
        """
        Initializes the DualSOM wrapper and its internal mathematical engine.
        Automatically calculates the optimal map grid size based on data volume.

        Args:
            parameters (dict): The hyperparameter dictionary loaded from JSON.
            coded_data (tuple): A tuple containing (X_features, y_labels).
        """
        self.parameters = parameters
        self.run_mode = parameters['run_mode']

        # IO Parameters for model persistence
        self.load_model = parameters['som_load_model']
        self.model_path = parameters['som_model_path']

        X, _ = coded_data

        # Calculate the optimal grid size internally
        som_size_index = parameters['som_size_index']
        self.grid_size = thumb_rule(len(X), som_size_index)
        print(f"   [SOM Init] Automatically determined Grid Size: {self.grid_size}x{self.grid_size}")

        # Initialize the base SOM with parameters fetched from JSON
        self.som = BaseDualSom(
            self.grid_size, self.grid_size, input_len=X.shape[1],
            sigma=parameters['som_sigma'],
            sigma_target=parameters['som_sigma_target'],
            learning_rate=parameters['som_lr'],
            lr_target=parameters['som_lr_target'],
            activation_distance=parameters['activation_distance']
        )
        self.clusterer = None
        self._winmap = None
        self._train_labels = None

    def fit(self, coded_data):
        """
        Trains the SOM weights using the latent data representations.

        If `som_load_model` is enabled in the configuration, this method will bypass
        training and load the pre-computed NumPy weight matrix directly from disk.

        Args:
            coded_data (tuple): A tuple containing (X_train, y_train).

        Raises:
            FileNotFoundError: If `som_load_model` is True but the file does not exist.
        """
        X, y = coded_data
        self._train_labels = y

        if self.load_model:
            if os.path.exists(self.model_path):
                print(f"\n>>> Loading pre-trained SOM weights from {self.model_path}...")
                self.som._weights = np.load(self.model_path)
            else:
                raise FileNotFoundError(
                    f"JSON config requested 'som_load_model': true, "
                    f"but no model found at {self.model_path}"
                )
        else:
            print("\n>>> Training SOM from scratch...")
            # Initialize weights utilizing PCA to accelerate topological convergence
            self.som.pca_weights_init(X)

            max_iter = self.parameters['som_epochs'] * len(X)

            # Short-circuit validation to save time if operating in unsupervised mode
            enable_val = bool(self.parameters['som_enable_validation'])
            if self.run_mode == 'unsupervised':
                enable_val = False

            self.som.train_batch(
                X, y, X, y, max_iter, verbose=True, enable_validation=enable_val
            )

            # Serialize the trained map weights for future rapid loading
            os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)
            np.save(self.model_path, self.som._weights)
            print(f">>> SOM weights saved successfully to {self.model_path}")

    def predict(self, coded_data, mode='clustering'):
        """
        Maps data to the SOM grid and assigns final predictions based on the mode.

        Args:
            coded_data (tuple): A tuple containing (X_data, y_labels).
            mode (str, optional): Determines the logic flow ('clustering' or 'classification').
                                  Defaults to 'clustering'.

        Returns:
            np.ndarray: A 1D array of predicted class labels or cluster IDs.
        """
        X, _ = coded_data

        if mode == 'clustering':
            # Initialize and fit the weight-space K-Means clusterer if not already trained
            if self.clusterer is None:
                n_clusters = self.parameters['n_clusters']
                max_iter = self.parameters['kmeans_max_iter']
                threshold = self.parameters['kmeans_threshold']

                self.clusterer = SOMClusterer(n_clusters=n_clusters, max_iter=max_iter, threshold=threshold)
                self.clusterer.fit(self.som._weights)

            # Prediction routes data to BMU, then assigns the BMU's cluster ID to the data
            return self.clusterer.predict(self.som, X)

        elif mode == 'classification':
            # Construct the BMU label voting map if not already built
            if self._winmap is None:
                self._winmap = self.som.labels_map(X, self._train_labels)

            # Establish a global fallback class for edge-cases where a test sample hits a virgin neuron
            default_class = sum(self._winmap.values(), Counter()).most_common()[0][0]
            result = []

            for d in X:
                win_pos = self.som.winner(d)
                if win_pos in self._winmap:
                    result.append(self._winmap[win_pos].most_common()[0][0])
                else:
                    result.append(default_class)

            return np.array(result)


class SOMClusterer:
    """
    A modified K-Means clustering algorithm adapted specifically for grouping
    Self-Organizing Map neuron weight vectors.

    It utilizes angular distance matrices to handle directional/skeletal latent
    features more effectively than standard Euclidean distance.

    Attributes:
        K (int): The target number of clusters.
        max_iter (int): Maximum algorithm iterations.
        threshold (float): Early stopping threshold.
        labels_map (np.ndarray): 2D grid mapping each SOM neuron to a cluster ID.
        centroids (np.ndarray): The learned geometric centers of the clusters.
    """

    def __init__(self, n_clusters, max_iter, threshold):
        """
        Initializes the clusterer with stopping criteria.

        Args:
            n_clusters (int): The target number of clusters (K).
            max_iter (int): Maximum iterations to prevent infinite loops.
            threshold (float): Early stopping threshold based on centroid displacement.
        """
        self.K = n_clusters
        self.max_iter = max_iter
        self.threshold = threshold
        self.labels_map = None
        self.centroids = None

    def _angular_distance_matrix(self, W1, W2):
        """
        Computes the pairwise angular distance matrix between two sets of vectors.

        Args:
            W1 (np.ndarray): First matrix of shape (N, D).
            W2 (np.ndarray): Second matrix of shape (M, D).

        Returns:
            np.ndarray: A distance matrix of shape (N, M).
        """
        W1_norm = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-8)
        W2_norm = W2 / (np.linalg.norm(W2, axis=1, keepdims=True) + 1e-8)
        sim = np.clip(np.dot(W1_norm, W2_norm.T), -1.0, 1.0)
        ang = np.arccos(sim) * 180 / np.pi
        ang = np.minimum(ang, 180 - ang)
        return ang / 180 * np.pi

    def fit(self, som_weights):
        """
        Executes the specialized K-Means algorithm over the converged SOM weight grid.

        Uses a farthest-first traversal strategy for robust initial centroid selection,
        followed by iterative centroid updates using the angular distance metric.

        Args:
            som_weights (np.ndarray): The 3D tensor representing the SOM weights
                                      with shape (width, height, features).

        Returns:
            SOMClusterer: The fitted instance of the clusterer.
        """
        grid_x, grid_y, features = som_weights.shape
        U = grid_x * grid_y
        vw = som_weights.reshape(U, features)
        dis = self._angular_distance_matrix(vw, vw)
        np.fill_diagonal(dis, 0)

        # Farthest-first traversal strategy for robust initial centroid selection
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
                break

        # Reshape the flat labels back to the 2D spatial format of the SOM
        self.labels_map = labels.reshape(grid_x, grid_y)
        return self

    def predict(self, som, data_X):
        """
        Projects data onto the SOM and maps them to their respective weight-space cluster.

        Args:
            som (BaseDualSom): The trained underlying SOM instance.
            data_X (np.ndarray): The input data matrix to cluster.

        Returns:
            np.ndarray: An array of predicted cluster indices for each input sample.
        """
        preds = []
        for x in data_X:
            win_x, win_y = som.winner(x)
            preds.append(self.labels_map[win_x, win_y])
        return np.array(preds)


class BaseDualSom(object):
    """
    Low-level mathematical implementation of the Self-Organizing Map.

    Augmented with dynamic distance routing (Euclidean, Cosine, Angular) and
    an exponential attention mechanism for robust weight updates.

    Attributes:
        _weights (np.ndarray): A 3D tensor holding the topological weights.
        _activation_map (np.ndarray): A 2D matrix capturing distances from input to nodes.
        _xx, _yy (np.ndarray): Meshgrid coordinates for spatial neighborhood calculations.
    """

    def __init__(self, x, y, input_len, sigma, sigma_target,
                 learning_rate, lr_target, activation_distance):
        """
        Initializes the SOM grid and configures update hyper-parameters.

        Args:
            x (int): Grid width in number of neurons.
            y (int): Grid height in number of neurons.
            input_len (int): Dimensionality of the input feature space.
            sigma (float): Initial neighborhood radius.
            sigma_target (float): Asymptotic limit for radius decay.
            learning_rate (float): Initial learning rate (alpha).
            lr_target (float): Asymptotic limit for learning rate decay.
            activation_distance (str): Metric used to compute node activations
                                       ('angular', 'euclidean', 'cosine').

        Raises:
            ValueError: If an unsupported `activation_distance` is provided.
        """
        self._learning_rate = learning_rate
        self._lr_target = lr_target
        self._sigma = sigma
        self._sigma_target = sigma_target
        self._input_len = input_len

        # Initialization of map weights on a unit hypersphere
        self._weights = np.random.rand(x, y, input_len) * 2 - 1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = np.zeros((x, y))
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
            msg = (f"'{activation_distance}' not supported. "
                   f"Distances available: {', '.join(distance_functions.keys())}")
            raise ValueError(msg)

        self._activation_distance = distance_functions[activation_distance]

    def _euclidean_distance(self, x, w):
        """
        Computes standard L2 (Euclidean) distance between input and weights.

        Args:
            x (np.ndarray): The input vector.
            w (np.ndarray): The weight tensor.

        Returns:
            np.ndarray: The 2D distance map.
        """
        return linalg.norm(np.subtract(x, w), axis=-1)

    def _cosine_distance(self, x, w):
        """
        Computes the Cosine distance metric (1 - Cosine Similarity).

        Args:
            x (np.ndarray): The input vector.
            w (np.ndarray): The weight tensor.

        Returns:
            np.ndarray: The 2D distance map.
        """
        num = (w * x).sum(axis=2)
        denum = np.multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum + 1e-8)

    def _angular_distance(self, x, w):
        """
        Computes the Angular distance metric derived from cosine similarity.
        Maps the similarity scalar directly to an angular value in radians [0, pi].

        Args:
            x (np.ndarray): The input vector.
            w (np.ndarray): The weight tensor.

        Returns:
            np.ndarray: The 2D distance map.
        """
        num = (w * x).sum(axis=2)
        denum = np.multiply(linalg.norm(w, axis=2), linalg.norm(x))
        res = num / (denum + 1e-8)
        res = np.clip(res, -1.0, 1.0)
        i = np.arccos(res) * 180 / np.pi
        tt = 180 - i
        res = np.minimum(i, tt)
        return res / 180 * np.pi

    def _activate(self, x):
        """
        Computes the activation distance map across the entire grid for a given input.

        Args:
            x (np.ndarray): A single input sample vector.
        """
        self._activation_map = self._activation_distance(x, self._weights)

    def winner(self, x):
        """
        Identifies the coordinates of the Best Matching Unit (BMU).

        Args:
            x (np.ndarray): A single input sample vector.

        Returns:
            tuple: A tuple containing the (x, y) grid coordinates of the BMU.
        """
        self._activate(x)
        return np.unravel_index(self._activation_map.argmin(), self._activation_map.shape)

    def update(self, x, win, t, max_iteration):
        """
        Updates the synaptic weights of the map towards the given input sample.

        Incorporates an exponential attention mechanism where the standard
        neighborhood update is penalized heavily if the node's activation
        distance to the input is exceedingly large.

        Args:
            x (np.ndarray): The input sample vector.
            win (tuple): The (x, y) coordinates of the BMU.
            t (int): Current iteration step.
            max_iteration (int): Total number of planned iterations.
        """
        tt = t / max_iteration

        # Apply exponential decay to learning rate and neighborhood radius
        eta = self._learning_rate * (self._lr_target / self._learning_rate) ** tt
        sig = self._sigma * (self._sigma_target / self._sigma) ** tt

        # Compute the spatial neighborhood mask around the BMU
        ax = np.logical_and(self._neigx > win[0] - sig, self._neigx < win[0] + sig)
        ay = np.logical_and(self._neigy > win[1] - sig, self._neigy < win[1] + sig)
        g = outer(ax, ay) * 1.0 * eta

        # Apply exponential attention decay based on topological activation map
        gg = self._activation_map.copy()
        G = einsum('ij, ij->ij', g, np.exp(-gg + np.pi))

        # Perform the vectorized competitive weight update
        self._weights += einsum('ij, ijk->ijk', G, x - self._weights)

    def pca_weights_init(self, data):
        """
        Initializes the SOM weights to span the first two principal components.

        This alignment rapidly accelerates topological convergence by ensuring the
        initial map unfolds along the highest variance directions of the dataset.

        Args:
            data (np.ndarray): The full training dataset.
        """
        pc_length, pc = linalg.eig(np.cov(np.transpose(data)))
        pc_order = np.argsort(-pc_length)
        for i, c1 in enumerate(np.linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(np.linspace(-1, 1, len(self._neigy))):
                self._weights[i, j] = c1 * pc[:, pc_order[0]] + c2 * pc[:, pc_order[1]]

    def train_batch(self, data, Y_train=None, X_test=None, Y_test=None,
                    max_iter=1000, verbose=False, enable_validation=True):
        """
        The main training loop executed over the generated sequence of iterations.

        Includes an optional mechanism for periodic accuracy validation, ensuring
        insight into the training process without evaluating every single step.

        Args:
            data (np.ndarray): The training input features.
            Y_train (np.ndarray, optional): Training ground truth labels.
            X_test (np.ndarray, optional): Validation input features.
            Y_test (np.ndarray, optional): Validation ground truth labels.
            max_iter (int, optional): The total number of iterations. Defaults to 1000.
            verbose (bool, optional): Controls output logging. Defaults to False.
            enable_validation (bool, optional): Toggles periodic accuracy reports.
                                                Defaults to True.
        """
        iterations = np.arange(max_iter) % len(data)

        idx = 1
        for t, iteration in enumerate(tqdm(iterations, desc="SOM Training", unit="iter")):

            # Trigger periodic accuracy validations (primarily applicable to supervised mode)
            if enable_validation and t == idx * len(data) * 5:
                winmap = self.labels_map(data, Y_train)
                default_class = Counter(Y_train).most_common(1)[0][0]
                result = []
                for d in X_test:
                    win_pos = self.winner(d)
                    if win_pos in winmap:
                        result.append(winmap[win_pos].most_common()[0][0])
                    else:
                        result.append(default_class)

                acc = round(accuracy_score(Y_test, result) * 100, 2)
                tqdm.write(f"Validation at step {idx}: Accuracy {acc}%")
                idx += 1

            self.update(data[iteration], self.winner(data[iteration]), t, max_iter)

    def labels_map(self, data, labels):
        """
        Constructs a voting map associating each neuron with class labels.

        Projects the entire dataset onto the grid and groups the ground truth labels
        of all samples that share the same Best Matching Unit (BMU).

        Args:
            data (np.ndarray): The dataset used to populate the map.
            labels (np.ndarray): The corresponding ground truth labels.

        Returns:
            defaultdict: A dictionary mapping grid coordinates (tuple) to a
                         `Counter` object representing the label distribution
                         for that specific neuron.
        """
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.winner(x)].append(l)

        for pos in winmap:
            winmap[pos] = Counter(winmap[pos])

        return winmap
