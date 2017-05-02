try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    pass

import numpy as np

import cupy


class KMeans(object):
    def __init__(self, n_clusters=2, max_iter=10, elem=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.elem = elem
        self.centers = None
        self.pred = None

    def fit(self, X):
        xp = cupy.get_array_module(X)
        self.pred = xp.zeros(len(X), dtype=np.int32)
        initial_indexes = np.random.choice(len(X), self.n_clusters,
                                           replace=False).astype(np.int32)
        self.centers = X[initial_indexes]
        data_num = X.shape[0]
        data_dim = X.shape[1]

        for _ in range(self.max_iter):
            # calculate distances and label
            distances = xp.zeros((data_num, self.n_clusters), dtype=np.float32)
            if self.elem is False or xp == np:
                for i in range(self.n_clusters):
                    distances[:, i] = xp.sum((X - self.centers[i]) ** 2,
                                             axis=1)
            else:
                cupy.ElementwiseKernel(
                    'S data, raw S centers', 'raw S dist',
                    '''
                    int cent_ind1[] = {0, i % 1000};
                    int cent_ind2[] = {1, i % 1000};
                    int dist_ind1[] = {i / 1000, 0};
                    int dist_ind2[] = {i / 1000, 1};
                    double diff1 = centers[cent_ind1] - data;
                    double diff2 = centers[cent_ind2] - data;
                    atomicAdd(&dist[dist_ind1], diff1 * diff1);
                    atomicAdd(&dist[dist_ind2], diff2 * diff2);
                    ''',
                    'calc_distances'
                )(X, self.centers, distances)

            new_pred = xp.argmin(distances, axis=1).astype(np.int32)
            if xp.all(new_pred == self.pred):
                break
            self.pred = new_pred

            # calculate centers
            if self.elem is False or xp == np:
                centers = xp.empty((0, data_dim), dtype=np.float32)
                for i in range(self.n_clusters):
                    centers = xp.vstack((centers,
                                         X[self.pred == i].mean(axis=0)))
            else:
                centers = xp.zeros((self.n_clusters, data_dim),
                                   dtype=np.float32)
                group = xp.zeros(self.n_clusters, dtype=np.float32)
                label = self.pred[:, None]
                cupy.ElementwiseKernel(
                    'S data, T label', 'raw S centers, raw S group',
                    '''
                    int cent_ind[] = {label, i % 1000};
                    atomicAdd(&centers[cent_ind], data);
                    atomicAdd(&group[label], 1);
                    ''',
                    'calc_center'
                )(X, label, centers, group)
                group /= data_dim
                centers /= group[:, None]

            self.centers = centers

    def draw(self, X, output):
        xp = cupy.get_array_module(X)
        for i in range(self.n_clusters):
            labels = X[self.pred == i]
            if xp == cupy:
                labels = labels.get()
                centers = self.centers.get()
            plt.scatter(labels[:, 0], labels[:, 1], color=np.random.rand(3, 1))
        plt.scatter(centers[:, 0], centers[:, 1], s=120, marker='s',
                    facecolors='y', edgecolors='k')
        plt.savefig(output + '.png')
