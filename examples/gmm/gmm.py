import matplotlib
matplotlib.use('Agg')
from matplotlib import mlab
import matplotlib.pyplot as plt
import six

import numpy as np

import cupy


class GaussianMixture(object):

    def __init__(self, tol=1e-3, reg_covar=1e-6, max_iter=100):
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.xp = None

    def estimate_gaussian_parameters(self, X, resp, reg_covar):
        nk = self.xp.sum(resp, axis=0)
        means = self.xp.dot(resp.T, X) / nk[:, None]
        avg_X2 = self.xp.dot(resp.T, X * X) / nk[:, None]
        avg_X_means = means * self.xp.dot(resp.T, X) / nk[:, None]
        covariances = avg_X2 - 2 * avg_X_means + means ** 2 + reg_covar
        return nk / len(X), means, covariances

    def estimate_log_prob(self, X, inv_cov):
        n_features = X.shape[1]
        det = self.xp.sum(self.xp.log(inv_cov), axis=1)
        precisions = inv_cov ** 2
        log_prob = self.xp.sum((self.means ** 2 * precisions), 1) - \
            2 * self.xp.dot(X, (self.means * precisions).T) + \
            self.xp.dot(X ** 2, precisions.T)
        return -0.5 * (n_features * self.xp.log(2 * np.pi) + log_prob) + det

    def e_step(self, X):
        weighted_log_prob = self.estimate_log_prob(X, self.inv_cov) \
            + self.xp.log(self.weights)
        log_prob_norm = self.xp.log(self.xp.sum(self.xp.exp(weighted_log_prob),
                                                axis=1))
        log_resp = weighted_log_prob - log_prob_norm[:, None]
        return self.xp.mean(log_prob_norm), log_resp

    def m_step(self, X, log_resp):
        self.weights, self.means, self.covariances = \
            self.estimate_gaussian_parameters(X, self.xp.exp(log_resp),
                                              self.reg_covar)
        self.inv_cov = 1 / self.xp.sqrt(self.covariances)

    def train_gmm(self, X):
        self.xp = cupy.get_array_module(X)
        lower_bound = -np.infty
        converged = False
        self.weights = self.xp.array([0.5, 0.5], dtype=np.float32)
        mean1 = self.xp.random.normal(3, self.xp.array([1, 2]), size=2)
        mean2 = self.xp.random.normal(-3, self.xp.array([2, 1]), size=2)
        self.means = self.xp.stack((mean1, mean2))
        self.covariances = self.xp.random.rand(2, 2)
        self.inv_cov = 1 / self.xp.sqrt(self.covariances)

        for n_iter in six.moves.range(self.max_iter):
            prev_lower_bound = lower_bound
            log_prob_norm, log_resp = self.e_step(X)
            self.m_step(X, log_resp)
            lower_bound = log_prob_norm
            change = lower_bound - prev_lower_bound
            if abs(change) < self.tol:
                converged = True
                break

        if not converged:
            msg = 'Failed to converge. Try different init parameters, \
                   or increase max_iter, tol or check for degenerate data.'
            print(msg)

    def predict(self, X):
        log_prob = self.estimate_log_prob(X, self.inv_cov)
        return (log_prob + self.xp.log(self.weights)).argmax(axis=1)

    def draw(self, X, pred, output):
        xp = cupy.get_array_module(X)
        for i in six.moves.range(2):
            labels = X[pred == i]
            if xp == cupy:
                labels = labels.get()
            plt.scatter(labels[:, 0], labels[:, 1], color=np.random.rand(3, 1))
        if xp == cupy:
            self.means = self.means.get()
            self.covariances = self.covariances.get()
        plt.scatter(self.means[:, 0], self.means[:, 1], s=120, marker='s',
                    facecolors='y', edgecolors='k')
        x = np.linspace(-10, 10, 1000)
        y = np.linspace(-10, 10, 1000)
        X, Y = np.meshgrid(x, y)
        for i in six.moves.range(2):
            Z = mlab.bivariate_normal(X, Y,
                                      np.sqrt(self.covariances[i][0]),
                                      np.sqrt(self.covariances[i][1]),
                                      self.means[i][0], self.means[i][1])
            plt.contour(X, Y, Z)
        plt.savefig(output + '.png')
