import argparse
import contextlib
import time

import cupy
import matplotlib.pyplot as plt
import numpy


@contextlib.contextmanager
def timer(message):
    cupy.cuda.Stream.null.synchronize()
    start = time.time()
    yield
    cupy.cuda.Stream.null.synchronize()
    end = time.time()
    print('%s:  %f sec' % (message, end - start))


var_kernel = cupy.ElementwiseKernel(
    'T x0, T x1, T c0, T c1', 'T out',
    'out = (x0 - c0) * (x0 - c0) + (x1 - c1) * (x1 - c1)',
    'var_kernel'
)
sum_kernel = cupy.ReductionKernel(
    'T x, S mask', 'T out',
    'mask ? x : 0',
    'a + b', 'out = a', '0',
    'sum_kernel'
)
count_kernel = cupy.ReductionKernel(
    'T mask', 'float32 out',
    'mask ? 1.0 : 0.0',
    'a + b', 'out = a', '0.0',
    'count_kernel'
)


def fit_xp(X, n_clusters, max_iter):
    assert X.ndim == 2

    # Get NumPy or CuPy module from the supplied array.
    xp = cupy.get_array_module(X)

    n_samples = len(X)

    # Make an array to store the labels indicating which cluster each sample is
    # contained.
    pred = xp.zeros(n_samples)

    # Choose the initial centroid for each cluster.
    initial_indexes = xp.random.choice(n_samples, n_clusters, replace=False)
    centers = X[initial_indexes]

    for _ in range(max_iter):
        # Compute the new label for each sample.
        distances = xp.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_pred = xp.argmin(distances, axis=1)

        # If the label is not changed for each sample, we suppose the
        # algorithm has converged and exit from the loop.
        if xp.all(new_pred == pred):
            break
        pred = new_pred

        # Compute the new centroid for each cluster.
        i = xp.arange(n_clusters)
        mask = pred == i[:, None]
        sums = xp.where(mask[:, :, None], X, 0).sum(axis=1)
        counts = xp.count_nonzero(mask, axis=1).reshape((n_clusters, 1))
        centers = sums / counts

    return centers, pred


def fit_custom(X, n_clusters, max_iter):
    assert X.ndim == 2

    n_samples = len(X)

    pred = cupy.zeros(n_samples)

    initial_indexes = cupy.random.choice(n_samples, n_clusters, replace=False)
    centers = X[initial_indexes]

    for _ in range(max_iter):
        distances = var_kernel(X[:, None, 0], X[:, None, 1],
                               centers[None, :, 1], centers[None, :, 0])
        new_pred = cupy.argmin(distances, axis=1)
        if cupy.all(new_pred == pred):
            break
        pred = new_pred

        i = cupy.arange(n_clusters)
        mask = pred == i[:, None]
        sums = sum_kernel(X, mask[:, :, None], axis=1)
        counts = count_kernel(mask, axis=1).reshape((n_clusters, 1))
        centers = sums / counts

    return centers, pred


def draw(X, n_clusters, centers, pred, output):
    # Plot the samples and centroids of the fitted clusters into an image file.
    for i in range(n_clusters):
        labels = X[pred == i]
        plt.scatter(labels[:, 0], labels[:, 1], c=numpy.random.rand(3))
    plt.scatter(
        centers[:, 0], centers[:, 1], s=120, marker='s', facecolors='y',
        edgecolors='k')
    plt.savefig(output)


def run(gpuid, n_clusters, num, max_iter, use_custom_kernel, output):
    samples = numpy.random.randn(num, 2)
    X_train = numpy.r_[samples + 1, samples - 1]

    with timer(' CPU '):
        centers, pred = fit_xp(X_train, n_clusters, max_iter)

    with cupy.cuda.Device(gpuid):
        X_train = cupy.asarray(X_train)

        with timer(' GPU '):
            if use_custom_kernel:
                centers, pred = fit_custom(X_train, n_clusters, max_iter)
            else:
                centers, pred = fit_xp(X_train, n_clusters, max_iter)

        if output is not None:
            index = numpy.random.choice(10000000, 300, replace=False)
            draw(X_train[index].get(), n_clusters, centers.get(),
                 pred[index].get(), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', '-g', default=0, type=int,
                        help='ID of GPU.')
    parser.add_argument('--n-clusters', '-n', default=2, type=int,
                        help='number of clusters')
    parser.add_argument('--num', default=5000000, type=int,
                        help='number of samples')
    parser.add_argument('--max-iter', '-m', default=10, type=int,
                        help='number of iterations')
    parser.add_argument('--use-custom-kernel', action='store_true',
                        default=False, help='use Elementwise kernel')
    parser.add_argument('--output-image', '-o', default=None, type=str,
                        help='output image file name')
    args = parser.parse_args()
    run(args.gpu_id, args.n_clusters, args.num, args.max_iter,
        args.use_custom_kernel, args.output_image)
