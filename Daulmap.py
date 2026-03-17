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
    (Cleaned & Optimized Version)
"""


def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None,
                             use_epochs=False):
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training."""
    if use_epochs:
        iterations_per_epoch = arange(data_len)
        if random_generator:
            random_generator.shuffle(iterations_per_epoch)
        iterations = tile(iterations_per_epoch, num_iterations)
    else:
        iterations = arange(num_iterations) % data_len
        if random_generator:
            random_generator.shuffle(iterations)
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations


def _wrap_index__in_verbose(iterations):
    """Yields the values in iterations printing the status on the stdout."""
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m - i + 1) * (time() - beginning)) / (i + 1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i + 1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100 * (i + 1) / m)
        progress += ' - {time_left} left '
        stdout.write(progress)


def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process."""
    return learning_rate / (1 + t / (max_iter / 2))


def new_decay(learning_rate, target, t, max_iter):
    """Custom exponential decay function."""
    tt = t / max_iter
    return learning_rate * (target / learning_rate) ** tt


class DualSom(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, repulsion=1,
                 decay_function=asymptotic_decay,
                 neighborhood_function='bubble', topology='rectangular',
                 activation_distance='angular', random_seed=None, encoder=None):
        """
        Initializes a Self Organizing Maps.
        """
        self.labelencoder = encoder
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = random.RandomState(random_seed)
        self._repulsion = repulsion
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len

        # Random initialization
        self._weights = self._random_generator.rand(x, y, input_len) * 2 - 1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        # [修复] 修正了拼写错误 acuracy -> accuracy
        self.accuracy = []
        self.it = []
        self._activation_map = zeros((x, y))
        self._neigx = arange(x)
        self._neigy = arange(y)

        if topology not in ['hexagonal', 'rectangular']:
            msg = '%s not supported only hexagonal and rectangular available'
            raise ValueError(msg % topology)
        self.topology = topology
        self._xx, self._yy = meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
        if topology == 'hexagonal':
            self._xx[::-2] -= 0.5

        self._decay_function = decay_function

        neig_functions = {'bubble': self._bubble,
                          'gaussian': self._gaussian}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['bubble'] and (divmod(sigma, 1)[1] != 0 or sigma < 1):
            warn('sigma should be an integer >=1 when bubble are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]

        distance_functions = {"angular": self._angular_distance,
                              "euclidean": self._euclidean_distance,
                              "cosine": self._cosine_distance}

        if isinstance(activation_distance, str):
            if activation_distance not in distance_functions:
                msg = '%s not supported. Distances available: %s'
                raise ValueError(msg % (activation_distance,
                                        ', '.join(distance_functions.keys())))
            self._activation_distance = distance_functions[activation_distance]
        elif callable(activation_distance):
            self._activation_distance = activation_distance

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        self._activation_map = self._activation_distance(x, self._weights)

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma."""
        ax = logical_and(self._neigx > c[0] - sigma,
                         self._neigx < c[0] + sigma)
        ay = logical_and(self._neigy > c[1] - sigma,
                         self._neigy < c[1] + sigma)
        return outer(ax, ay) * 1.

    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2 * sigma * sigma
        ax = exp(-power(self._xx - self._xx.T[c], 2) / d)
        ay = exp(-power(self._yy - self._yy.T[c], 2) / d)
        return (ax * ay).T

    def _euclidean_distance(self, x, w):
        return linalg.norm(subtract(x, w), axis=-1)

    def _cosine_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum + 1e-8)

    def _angular_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        res = num / (denum + 1e-8)

        res = np.clip(res, -1.0, 1.0)

        i = np.arccos(res)
        i = i / np.pi * 180
        tt = 180 - i
        res = np.minimum(i, tt)
        return res / 180 * np.pi

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)

    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons."""
        eta = new_decay(self._learning_rate, 0.001, t, max_iteration)
        sig = new_decay(self._sigma, 0.01, t, max_iteration)

        np.set_printoptions(threshold=np.inf)

        g = self.neighborhood(win, sig) * eta
        dif = self._activation_map
        gg = dif.copy()

        # Custom attention mechanism
        G = einsum('ij, ij->ij', g, exp(-gg + np.pi))

        self._weights += einsum('ij, ijk->ijk', G, x - self._weights)

    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components."""
        if self._input_len == 1:
            msg = 'The data needs at least 2 features for pca initialization'
            raise ValueError(msg)
        self._check_input_len(data)
        if len(self._neigx) == 1 or len(self._neigy) == 1:
            msg = 'PCA initialization inappropriate: One of the dimensions of the map is 1.'
            warn(msg)

        pc_length, pc = linalg.eig(cov(transpose(data)))
        pc_order = argsort(-pc_length)
        for i, c1 in enumerate(linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(linspace(-1, 1, len(self._neigy))):
                self._weights[i, j] = c1 * pc[:, pc_order[0]] + \
                                      c2 * pc[:, pc_order[1]]

    def train(self, data, *args, random_order=False, verbose=False, use_epochs=False, enable_validation=True):
        """Trains the SOM. Supports both complete and minimal arguments."""

        if len(args) == 1:
            Y_train, X_test, Y_test = None, None, None
            num_iteration = args[0]
        elif len(args) == 4:
            Y_train, X_test, Y_test, num_iteration = args
            # [修复] 确保输入数据是 numpy 数组，防止索引/切片报错
            X_test = np.asarray(X_test)
            Y_test = np.asarray(Y_test)
            Y_train = np.asarray(Y_train)
        else:
            raise ValueError(
                "Invalid arguments. Use train(data, num_iteration) OR train(data, Y_train, X_test, Y_test, num_iteration)")

        if enable_validation and (Y_train is None or X_test is None or Y_test is None):
            enable_validation = False

        def valid_accuracy():
            winmap = self.labels_map(data, Y_train)
            if not winmap:
                return 0.0

            # [修复] 移除了 $O(N^2)$ 的 Counter 累加瓶颈，直接对原数组统计频率
            default_class = Counter(Y_train).most_common(1)[0][0]

            result = []
            for d in X_test:
                win_position = self.winner(d)
                if win_position in winmap:
                    result.append(winmap[win_position].most_common()[0][0])
                else:
                    result.append(default_class)
            res = accuracy_score(Y_test, result)
            print(res, "acc")
            return res

        self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        random_generator = None
        if random_order:
            random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              verbose, random_generator,
                                              use_epochs)

        # [修复] 移除了错误的 get_decay_rate 计算，直接将 t 传给 update 方法
        idx = 1
        idy = 1
        for t, iteration in enumerate(iterations):
            if t == idy * len(data):
                print("epoch", idy)
                idy += 1
            if t == idx * len(data) * 5:
                if enable_validation:
                    acc = round(valid_accuracy() * 100, 2)
                    self.accuracy.append(acc)
                    self.it.append(idx)
                    print(idx)
                idx += 1

            # 直接传入 t 作为进度，而不是固定的 epoch_index
            self.update(data[iteration], self.winner(data[iteration]),
                        t, num_iteration)

        if enable_validation:
            self.accuracy.append(round(valid_accuracy() * 100, 2))
            self.it.append(idx)
            print(idx)

    def train_batch(self, data, *args, verbose=False, enable_validation=True):
        """Trains the SOM using all the vectors in data sequentially."""
        self.train(data, *args, random_order=False, verbose=verbose, enable_validation=enable_validation)

    def labels_map(self, data, labels):
        """Returns a dictionary wm where wm[(i,j)] is a dictionary
        that contains the number of samples from a given label
        that have been mapped in position i,j."""
        self._check_input_len(data)
        if not len(data) == len(labels):
            raise ValueError('data and labels must have the same length.')
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.winner(x)].append(l)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap