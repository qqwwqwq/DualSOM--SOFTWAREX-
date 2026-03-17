import math
import numpy as np
from numpy import (array, unravel_index, nditer, linalg, random, subtract, max,
                   power, exp, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, cov, argsort, linspace, transpose,
                   einsum, prod, nan, sqrt, hstack, diff, argmin, multiply,
                   nanmean, nansum, tile, array_equal)
from collections import defaultdict, Counter
from warnings import warn
from sys import stdout
from time import time
from datetime import timedelta
from sklearn.metrics import accuracy_score

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
    A Dual-SOM extends the traditional SOM by incorporating advanced 
    distance metrics and an exponential attention mechanism for weight updates.
"""


def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None,
                             use_epochs=False):
    """
    Generates an array of indices determining the order in which data samples 
    are presented to the SOM during training.
    
    Args:
        data_len: Number of samples in the dataset.
        num_iterations: Total number of training steps.
        use_epochs: If True, each sample is seen once per epoch before repeating.
    """
    if use_epochs:
        # Create a full pass over the data indices
        iterations_per_epoch = arange(data_len)
        if random_generator:
            random_generator.shuffle(iterations_per_epoch)
        # Repeat the shuffled epoch multiple times
        iterations = tile(iterations_per_epoch, num_iterations)
    else:
        # Standard approach: Pick samples via modulo or random sampling
        iterations = arange(num_iterations) % data_len
        if random_generator:
            random_generator.shuffle(iterations)
            
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations


def _wrap_index__in_verbose(iterations):
    """
    A wrapper that yields iteration indices while printing a progress bar 
    and estimated time remaining (ETA) to the standard output.
    """
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    
    for i, it in enumerate(iterations):
        yield it
        # Calculate ETA based on average time per iteration
        sec_left = ((m - i + 1) * (time() - beginning)) / (i + 1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i + 1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100 * (i + 1) / m)
        progress += ' - {time_left} left '
        stdout.write(progress)


def asymptotic_decay(learning_rate, t, max_iter):
    """
    Standard 1/t decay. The learning rate reduces slowly over time 
    to allow the map to settle into a stable state.
    """
    return learning_rate / (1 + t / (max_iter / 2))


def new_decay(learning_rate, target, t, max_iter):
    """
    Exponential decay function. 
    Moves from initial 'learning_rate' to 'target' over 'max_iter' steps.
    """
    tt = t / max_iter
    return learning_rate * (target / learning_rate) ** tt


class DualSom(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, repulsion=1,
                 decay_function=asymptotic_decay,
                 neighborhood_function='bubble', topology='rectangular',
                 activation_distance='angular', random_seed=None, encoder=None):
        """
        Initializes the SOM grid.
        
        Args:
            x, y: Dimensions of the SOM grid (neurons).
            input_len: Dimensionality of the input features.
            sigma: Initial neighborhood radius.
            learning_rate: Initial step size for weight updates.
            activation_distance: Metric used to find the Best Matching Unit (BMU).
        """
        self.labelencoder = encoder
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = random.RandomState(random_seed)
        self._repulsion = repulsion
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len

        # Weight Initialization: Randomly distributed on a unit hypersphere
        self._weights = self._random_generator.rand(x, y, input_len) * 2 - 1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        # Track accuracy and iteration history for visualization/validation
        self.accuracy = []
        self.it = []
        self._activation_map = zeros((x, y))
        self._neigx = arange(x)
        self._neigy = arange(y)

        # Grid Topology Setup
        if topology not in ['hexagonal', 'rectangular']:
            msg = '%s not supported only hexagonal and rectangular available'
            raise ValueError(msg % topology)
        self.topology = topology
        
        # Create coordinates for the grid neurons
        self._xx, self._yy = meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
        
        # Hexagonal grids shift every other row to create 6 neighbors per neuron
        if topology == 'hexagonal':
            self._xx[::-2] -= 0.5

        self._decay_function = decay_function

        # Neighborhood functions define how neighbors of the BMU are updated
        neig_functions = {'bubble': self._bubble,
                          'gaussian': self._gaussian}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        self.neighborhood = neig_functions[neighborhood_function]

        # Distance metrics for finding the winner (BMU)
        distance_functions = {"angular": self._angular_distance,
                              "euclidean": self._euclidean_distance,
                              "cosine": self._cosine_distance}

        if isinstance(activation_distance, str):
            self._activation_distance = distance_functions[activation_distance]
        elif callable(activation_distance):
            self._activation_distance = activation_distance

    def _activate(self, x):
        """Calculates the distance between input x and every neuron in the grid."""
        self._activation_map = self._activation_distance(x, self._weights)

    def activate(self, x):
        """Public method to get the activation map."""
        self._activate(x)
        return self._activation_map

    def _bubble(self, c, sigma):
        """
        Bubble neighborhood: Returns a binary mask (square) where all neurons 
        within radius sigma of center c have a value of 1.0.
        """
        ax = logical_and(self._neigx > c[0] - sigma,
                         self._neigx < c[0] + sigma)
        ay = logical_and(self._neigy > c[1] - sigma,
                         self._neigy < c[1] + sigma)
        return outer(ax, ay) * 1.

    def _gaussian(self, c, sigma):
        """
        Gaussian neighborhood: Returns a smooth radial decay centered at c.
        Allows for more fine-grained updates than the bubble function.
        """
        d = 2 * sigma * sigma
        ax = exp(-power(self._xx - self._xx.T[c], 2) / d)
        ay = exp(-power(self._yy - self._yy.T[c], 2) / d)
        return (ax * ay).T

    def _euclidean_distance(self, x, w):
        """Standard L2 distance."""
        return linalg.norm(subtract(x, w), axis=-1)

    def _cosine_distance(self, x, w):
        """1 - Cosine Similarity. Useful for high-dimensional directional data."""
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum + 1e-8)

    def _angular_distance(self, x, w):
        """
        Angular distance derived from cosine similarity.
        Measures the angle between vectors, capping at 180 degrees.
        """
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        res = num / (denum + 1e-8)
        res = np.clip(res, -1.0, 1.0)
        i = np.arccos(res)
        i = i / np.pi * 180
        tt = 180 - i
        res = np.minimum(i, tt)
        return res / 180 * np.pi

    def winner(self, x):
        """Finds the (x, y) coordinates of the neuron closest to input x."""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)

    def update(self, x, win, t, max_iteration):
        """
        Updates the weights of the SOM neurons.
        Uses a Dual-SOM approach: standard neighborhood + an exponential attention 
        mechanism based on the activation distance.
        """
        # Eta (learning rate) and Sig (neighborhood radius) decay over time
        eta = new_decay(self._learning_rate, 0.001, t, max_iteration)
        sig = new_decay(self._sigma, 0.01, t, max_iteration)

        # Calculate standard neighborhood influence
        g = self.neighborhood(win, sig) * eta
        
        # Dual-SOM specific: The Attention Mechanism
        # Weight updates are further modulated by the distance (exp(-dist))
        # Neurons that are already 'far' from the sample get updated less.
        dif = self._activation_map
        G = einsum('ij, ij->ij', g, exp(-dif + np.pi))

        # Competitive Update: w = w + G * (x - w)
        self._weights += einsum('ij, ijk->ijk', G, x - self._weights)

    def pca_weights_init(self, data):
        """
        Initializes weights using the first two principal components of the data.
        This often leads to much faster convergence than random initialization.
        """
        self._check_input_len(data)
        # Compute the covariance matrix and eigenvalues/vectors
        pc_length, pc = linalg.eig(cov(transpose(data)))
        pc_order = argsort(-pc_length) # Order by variance
        
        # Map the 2D grid coordinates to the PC1-PC2 space
        for i, c1 in enumerate(linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(linspace(-1, 1, len(self._neigy))):
                self._weights[i, j] = c1 * pc[:, pc_order[0]] + \
                                      c2 * pc[:, pc_order[1]]

    def train(self, data, *args, random_order=False, verbose=False, use_epochs=False, enable_validation=True):
        """
        Main training loop. 
        Supports flexible arguments: train(data, iters) or train(data, y_train, x_test, y_test, iters).
        """
        if len(args) == 1:
            Y_train, X_test, Y_test = None, None, None
            num_iteration = args[0]
        elif len(args) == 4:
            Y_train, X_test, Y_test, num_iteration = args
            X_test, Y_test, Y_train = map(np.asarray, [X_test, Y_test, Y_train])
        else:
            raise ValueError("Invalid arguments.")

        if enable_validation and (Y_train is None or X_test is None or Y_test is None):
            enable_validation = False

        def valid_accuracy():
            """Evaluates the map by assigning class labels to neurons based on training data."""
            winmap = self.labels_map(data, Y_train)
            if not winmap: return 0.0

            # Default to the most common class in the training set for unmapped neurons
            default_class = Counter(Y_train).most_common(1)[0][0]
            result = []
            for d in X_test:
                win_position = self.winner(d)
                if win_position in winmap:
                    result.append(winmap[win_position].most_common()[0][0])
                else:
                    result.append(default_class)
            return accuracy_score(Y_test, result)

        self._check_input_len(data)
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              verbose, self._random_generator if random_order else None,
                                              use_epochs)

        # Training loop
        idx = 1
        for t, iteration in enumerate(iterations):
            # Periodic validation and status logging
            if t == idx * len(data) * 5:
                if enable_validation:
                    acc = round(valid_accuracy() * 100, 2)
                    self.accuracy.append(acc)
                    self.it.append(idx)
                    print(f"Validation at step {idx}: {acc}%")
                idx += 1

            # Update the map weights for the current sample
            self.update(data[iteration], self.winner(data[iteration]), t, num_iteration)

    def train_batch(self, data, *args, verbose=False, enable_validation=True):
        """Sequential training using the fixed order of the dataset."""
        self.train(data, *args, random_order=False, verbose=verbose, enable_validation=enable_validation)

    def labels_map(self, data, labels):
        """
        Creates a 'semantic' map where each neuron (i, j) is associated 
        with the labels of the training samples that mapped to it.
        Returns: { (i,j) : Counter({label1: count, label2: count}) }
        """
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.winner(x)].append(l)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap
