import numpy as np

import cupy


class KMeans(object):
    def __init__(self, n_clusters=2, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.xp = None
        self.pred = None

    def fit(self, X):
        self.xp = cupy.get_array_module(X)
        self.pred = self.xp.zeros(len(X))
        X_indexes = np.arange(len(X))
        np.random.shuffle(X_indexes)
        initial_centroid_indexes = X_indexes[:self.n_clusters]
        self.centers = X[initial_centroid_indexes]
        data_num = X.shape[0]
        data_dim = X.shape[1]

        for _ in range(self.max_iter):
            # calculate distances and label
            X2 = X.repeat(self.n_clusters, axis=0)
            X2 = X2.reshape(data_num, self.n_clusters, data_dim)
            centers = self.xp.broadcast_to(self.centers,
                                           (data_num, self.n_clusters,
                                            data_dim))
            new_pred = self.xp.argmin(self.xp.sum((X2 - centers) ** 2,
                                                  axis=2), axis=1)

            if self.xp.all(new_pred == self.pred):
                break
            self.pred = new_pred

            # calculate centers
            if self.xp == np:
                centers = self.xp.empty((0, data_dim))
                for i in range(self.n_clusters):
                    centers = self.xp.vstack((centers,
                                              X[self.pred == i].mean(axis=0)))
            else:
                centers = self.xp.zeros((self.n_clusters, data_dim),
                                        dtype=self.xp.float64)
                group = self.xp.zeros(self.n_clusters, dtype=self.xp.float64)
                label = self.xp.expand_dims(self.pred, axis=1)
                cupy.ElementwiseKernel(
                    'S data, T label', 'raw U centers, raw U group',
                    '''
                    int cent_ind[] = {label, i % 30000};
                    atomicAdd(&centers[cent_ind], data);
                    atomicAdd(&group[label], 1);
                    ''',
                    'calc_center'
                )(X, label, centers, group)
                group /= data_dim
                centers /= self.xp.expand_dims(group, axis=1)

            self.centers = centers

    def draw(self, X, output):
        if self.n_clusters > 8:
            raise ValueError("n_clusters have to be less than 8")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        color = ['r', 'b', 'g', 'c', 'm', 'k', 'w', 'y']
        for i in range(self.n_clusters):
            labels = X[self.pred == i]
            if self.xp == cupy:
                labels = self.xp.asnumpy(labels)
                self.centers = self.xp.asnumpy(self.centers)
            plt.scatter(labels[:, 0], labels[:, 1], color=color[i])
        plt.scatter(self.centers[:, 0], self.centers[:, 1], s=120, marker='s',
                    facecolors='y', edgecolors='k')
        plt.savefig(output + '.png')
