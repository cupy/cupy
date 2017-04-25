import numpy as np

import cupy


class KMeans(object):
    def __init__(self, n_clusters=2, max_iter=30, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
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
            X2 = X.repeat(self.n_clusters, axis=0)
            X2 = X2.reshape(data_num, self.n_clusters, data_dim)
            centers = self.xp.broadcast_to(self.centers,
                                           (data_num, self.n_clusters,
                                            data_dim))
            new_pred = self.xp.argmin(self.xp.sum((X2 - centers) ** 2,
                                                  axis=2), axis=1)

            # if np.all(new_pred == pred):
            #     break
            self.pred = new_pred

            centers = self.xp.empty((0, data_dim))
            for i in range(self.n_clusters):
                centers = self.xp.vstack((centers,
                                          X[self.pred == i].mean(axis=0)))
            self.centers = centers

    def draw(self, X, output):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for i in range(self.n_clusters):
            labels = X[self.pred == i]
            plt.scatter(labels[:, 0], labels[:, 1])
        plt.scatter(self.centers[:, 0], self.centers[:, 1], s=100,
                    facecolors='none', edgecolors='black')
        plt.savefig(output + '.png')
