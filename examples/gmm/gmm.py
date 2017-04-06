import numpy as np

import cupy


class GaussianMixture(object):

    def __init__(self, n_components=1, tol=1e-3, reg_covar=1e-6, max_iter=100,
                 seed=None):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.seed = seed
        self.xp = None

    def _compute_precision_cholesky(self, covariances):
        estimate_precision_error_message = (
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by "
            "singleton or collapsed samples). Try to decrease the "
            "number of components, or increase reg_covar.")

        if self.xp.any(self.xp.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / self.xp.sqrt(covariances)
        return precisions_chol

    def _estimate_gaussian_covariances(self, resp, X, nk, means, reg_covar):
        avg_X2 = self.xp.dot(resp.T, X * X) / nk[:, self.xp.newaxis]
        avg_means2 = means ** 2
        avg_X_means = means * self.xp.dot(resp.T, X) / nk[:, self.xp.newaxis]
        return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

    def _estimate_gaussian_parameters(self, X, resp, reg_covar):
        nk = resp.sum(axis=0) + 10 * self.xp.finfo(resp.dtype).eps
        means = self.xp.dot(resp.T, X) / nk[:, self.xp.newaxis]
        covariances = self._estimate_gaussian_covariances(resp, X, nk,
                                                          means, reg_covar)
        return nk, means, covariances

    def _initialize(self, X, resp, n_samples):
        weights, means, covariances = \
            self._estimate_gaussian_parameters(X, resp, self.reg_covar)
        weights /= n_samples
        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.precisions_cholesky_ = \
            self._compute_precision_cholesky(covariances)

    def initialize_parameters(self, X, n_samples, random_state):
        resp = random_state.rand(n_samples, self.n_components)
        resp /= resp.sum(axis=1)[:, self.xp.newaxis]
        self._initialize(X, resp, n_samples)

    def _estimate_log_gaussian_prob(self, X, means, precisions_chol):
        n_samples, n_features = X.shape
        n_components, _ = means.shape
        log_det = (self.xp.sum(self.xp.log(precisions_chol), axis=1))

        precisions = precisions_chol ** 2
        log_prob = (self.xp.sum((means ** 2 * precisions), 1) -
                    2. * self.xp.dot(X, (means * precisions).T) +
                    self.xp.dot(X ** 2, precisions.T))
        return -.5 * (n_features * self.xp.log(2 * np.pi) + log_prob) + log_det

    def _e_step(self, X):
        weighted_log_prob = self._estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_) + \
            self.xp.log(self.weights_)

        log_prob_norm = self.xp.log(self.xp.sum(self.xp.exp(weighted_log_prob),
                                                axis=1))
        with np.errstate(under='ignore'):
            log_resp = weighted_log_prob - log_prob_norm[:, self.xp.newaxis]

        return self.xp.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = (
            self._estimate_gaussian_parameters(X, self.xp.exp(log_resp),
                                               self.reg_covar))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = \
            self._compute_precision_cholesky(self.covariances_)

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_) = params

        # Attributes computation
        _, n_features = self.means_.shape

        self.precisions_ = self.precisions_cholesky_ ** 2

    def fit(self, X, y=None):
        self.xp = cupy.get_array_module(X)

        random_state = self.xp.random.RandomState(self.seed)
        max_lower_bound = -np.infty
        lower_bound = -np.infty
        converged = False
        n_samples = X.shape[0]

        self.initialize_parameters(X, n_samples, random_state)

        for n_iter in range(self.max_iter):
            prev_lower_bound = lower_bound

            log_prob_norm, log_resp = self._e_step(X)
            self._m_step(X, log_resp)
            lower_bound = log_prob_norm

            change = lower_bound - prev_lower_bound

            if abs(change) < self.tol:
                converged = True
                break

        if lower_bound > max_lower_bound:
            max_lower_bound = lower_bound
            best_params = (self.weights_, self.means_, self.covariances_,
                           self.precisions_cholesky_)
            best_n_iter = n_iter

        if not converged:
            msg = 'Failed to converge. Try different init parameters, \
                   or increase max_iter, tol or check for degenerate data.'
            print("%s" % msg)

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter

        return self

    def predict(self, X):
        result = self._estimate_log_gaussian_prob(X, self.means_,
                                                  self.precisions_cholesky_)
        return (result + self.xp.log(self.weights_)).argmax(axis=1)
