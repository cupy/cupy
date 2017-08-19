import argparse
import contextlib
import time

import matplotlib
matplotlib.use('Agg')
from matplotlib import mlab
import matplotlib.pyplot as plt
import numpy as np
import six

import cupy


@contextlib.contextmanager
def timer(message):
    cupy.cuda.Stream.null.synchronize()
    start = time.time()
    yield
    cupy.cuda.Stream.null.synchronize()
    end = time.time()
    print('%s:  %f sec' % (message, end - start))


def estimate_log_prob(X, inv_cov, means):
    xp = cupy.get_array_module(X)
    n_features = X.shape[1]
    log_det = xp.sum(xp.log(inv_cov), axis=1)
    precisions = inv_cov ** 2
    log_prob = xp.sum((means ** 2 * precisions), 1) - \
        2 * xp.dot(X, (means * precisions).T) + xp.dot(X ** 2, precisions.T)
    return -0.5 * (n_features * xp.log(2 * np.pi) + log_prob) + log_det


def m_step(X, resp):
    xp = cupy.get_array_module(X)
    nk = xp.sum(resp, axis=0)
    means = xp.dot(resp.T, X) / nk[:, None]
    X2 = xp.dot(resp.T, X * X) / nk[:, None]
    covariances = X2 - means ** 2
    return nk / len(X), means, covariances


def e_step(X, inv_cov, means, weights):
    xp = cupy.get_array_module(X)
    weighted_log_prob = estimate_log_prob(X, inv_cov, means) + \
        xp.log(weights)
    log_prob_norm = xp.log(xp.sum(xp.exp(weighted_log_prob), axis=1))
    log_resp = weighted_log_prob - log_prob_norm[:, None]
    return xp.mean(log_prob_norm), log_resp


def train_gmm(X, max_iter, tol, means, covariances):
    xp = cupy.get_array_module(X)
    lower_bound = -np.infty
    converged = False
    weights = xp.array([0.5, 0.5], dtype=np.float32)
    inv_cov = 1 / xp.sqrt(covariances)

    for n_iter in six.moves.range(max_iter):
        prev_lower_bound = lower_bound
        log_prob_norm, log_resp = e_step(X, inv_cov, means, weights)
        weights, means, covariances = m_step(X, xp.exp(log_resp))
        inv_cov = 1 / xp.sqrt(covariances)
        lower_bound = log_prob_norm
        change = lower_bound - prev_lower_bound
        if abs(change) < tol:
            converged = True
            break

    if not converged:
        print('Failed to converge. Increase max-iter or tol.')

    return inv_cov, means, weights, covariances


def predict(X, inv_cov, means, weights):
    xp = cupy.get_array_module(X)
    log_prob = estimate_log_prob(X, inv_cov, means)
    return (log_prob + xp.log(weights)).argmax(axis=1)


def calc_acc(X_train, y_train, X_test, y_test, max_iter, tol, means,
             covariances):
    xp = cupy.get_array_module(X_train)
    inv_cov, means, weights, cov = \
        train_gmm(X_train, max_iter, tol, means, covariances)
    y_train_pred = predict(X_train, inv_cov, means, weights)
    train_accuracy = xp.mean(y_train_pred == y_train) * 100
    y_test_pred = predict(X_test, inv_cov, means, weights)
    test_accuracy = xp.mean(y_test_pred == y_test) * 100
    print('train_accuracy : %f' % train_accuracy)
    print('test_accuracy : %f' % test_accuracy)
    return y_test_pred, means, cov


def draw(X, pred, means, covariances, output):
    xp = cupy.get_array_module(X)
    for i in six.moves.range(2):
        labels = X[pred == i]
        if xp is cupy:
            labels = labels.get()
        plt.scatter(labels[:, 0], labels[:, 1], c=np.random.rand(3))
    if xp is cupy:
        means = means.get()
        covariances = covariances.get()
    plt.scatter(means[:, 0], means[:, 1], s=120, marker='s', facecolors='y',
                edgecolors='k')
    x = np.linspace(-5, 5, 1000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    for i in six.moves.range(2):
        Z = mlab.bivariate_normal(X, Y, np.sqrt(covariances[i][0]),
                                  np.sqrt(covariances[i][1]),
                                  means[i][0], means[i][1])
        plt.contour(X, Y, Z)
    plt.savefig(output)


def run(gpuid, num, dim, max_iter, tol, output):
    '''CuPy Gaussian Mixture Model example

    Compute GMM parameters, weights, means and covariance matrix, depending on
    sampled data. There are two main components, e_step and m_step.
    In e_step, compute burden rate, which is expressed `resp`, by latest
    weights, means and covariance matrix.
    In m_step, compute weights, means and covariance matrix by latest `resp`.

    '''
    scale = np.ones(dim)
    train1 = np.random.normal(1, scale, size=(num, dim)).astype(np.float32)
    train2 = np.random.normal(-1, scale, size=(num, dim)).astype(np.float32)
    X_train = np.r_[train1, train2]
    test1 = np.random.normal(1, scale, size=(100, dim)).astype(np.float32)
    test2 = np.random.normal(-1, scale, size=(100, dim)).astype(np.float32)
    X_test = np.r_[test1, test2]
    y_train = np.r_[np.zeros(num), np.ones(num)].astype(np.int32)
    y_test = np.r_[np.zeros(100), np.ones(100)].astype(np.int32)

    mean1 = np.random.normal(1, scale, size=dim)
    mean2 = np.random.normal(-1, scale, size=dim)
    means = np.stack([mean1, mean2])
    covariances = np.random.rand(2, dim)
    print('Running CPU...')
    with timer(' CPU '):
        y_test_pred, means, cov = \
            calc_acc(X_train, y_train, X_test, y_test, max_iter, tol,
                     means, covariances)

    with cupy.cuda.Device(gpuid):
        X_train_gpu = cupy.array(X_train)
        y_train_gpu = cupy.array(y_train)
        y_test_gpu = cupy.array(y_test)
        X_test_gpu = cupy.array(X_test)
        means = cupy.array(means)
        covariances = cupy.array(covariances)
        print('Running GPU...')
        with timer(' GPU '):
            y_test_pred, means, cov = \
                calc_acc(X_train_gpu, y_train_gpu, X_test_gpu, y_test_gpu,
                         max_iter, tol, means, covariances)
        if output is not None:
            draw(X_test_gpu, y_test_pred, means, cov, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', '-g', default=0, type=int,
                        help='ID of GPU.')
    parser.add_argument('--num', '-n', default=500000, type=int,
                        help='number of train data')
    parser.add_argument('--dim', '-d', default=2, type=int,
                        help='dimension of each data')
    parser.add_argument('--max-iter', '-m', default=30, type=int,
                        help='number of iterations')
    parser.add_argument('--tol', '-t', default=1e-3, type=float,
                        help='error tolerance to stop iterations')
    parser.add_argument('--output-image', '-o', default=None, type=str,
                        dest='output', help='output image file name')
    args = parser.parse_args()
    run(args.gpu_id, args.num, args.dim, args.max_iter, args.tol, args.output)
