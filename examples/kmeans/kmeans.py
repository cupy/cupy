import argparse
import contextlib
import time

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


var_kernel = cupy.ElementwiseKernel(
    'T x0, T x1, T c0, T c1', 'T out',
    'out = (x0 - c0) * (x0 - c0) + (x1 - c1) * (x1 - c1)',
    'var_kernel'
)
sum_kernel = cupy.ReductionKernel(
    'T x, S p, S i', 'T out',
    'p == i ? x : 0',
    'a + b', 'out = a', '0',
    'sum_kernel'
)
count_kernel = cupy.ReductionKernel(
    'T p, S i', 'float32 out',
    'p == i ? 1.0 : 0.0',
    'a + b', 'out = a', '0.0',
    'count_kernel'
)


def fit(X, n_clusters, max_iter, use_custom_kernel):
    assert X.ndim == 2
    xp = cupy.get_array_module(X)
    pred = xp.zeros(len(X), dtype=np.int32)
    initial_indexes = np.random.choice(len(X), n_clusters,
                                       replace=False).astype(np.int32)
    centers = X[initial_indexes]

    for _ in six.moves.range(max_iter):
        # calculate distances and label
        if not use_custom_kernel or xp == np:
            distances = xp.linalg.norm(X[:, None, :] - centers[None, :, :],
                                       axis=2)
        else:
            distances = var_kernel(X[:, None, 0], X[:, None, 1],
                                   centers[None, :, 1], centers[None, :, 0])

        new_pred = xp.argmin(distances, axis=1).astype(np.int32)
        if xp.all(new_pred == pred):
            break
        pred = new_pred

        # calculate centers
        i = xp.arange(n_clusters, dtype=np.int32)
        if not use_custom_kernel or xp == np:
            mask = pred == i[:, None]
            sums = xp.where(mask[:, :, None], X, 0).sum(axis=1)
            counts = xp.count_nonzero(mask, axis=1)
            centers = sums / counts
        else:
            sums = sum_kernel(X, pred[:, None], i[:, None, None], axis=1)
            counts = count_kernel(pred, i[:, None], axis=1)
            centers = sums / counts

    return centers, pred


def draw(X, n_clusters, centers, pred, output):
    xp = cupy.get_array_module(X)
    for i in six.moves.range(n_clusters):
        labels = X[pred == i]
        if xp == cupy:
            labels = labels.get()
        plt.scatter(labels[:, 0], labels[:, 1], c=np.random.rand(3))
    if xp == cupy:
        centers = centers.get()
    plt.scatter(centers[:, 0], centers[:, 1], s=120, marker='s',
                facecolors='y', edgecolors='k')
    plt.savefig(output)


def run(gpuid, n_clusters, num, max_iter, use_custom_kernel, output):
    samples = np.random.randn(num, 2).astype(np.float32)
    X_train = np.r_[samples + 1, samples - 1]
    repeat = 1

    with timer(' CPU '):
        for i in range(repeat):
            centers, pred = fit(X_train, n_clusters, max_iter,
                                use_custom_kernel)

    with cupy.cuda.Device(gpuid):
        X_train = cupy.asarray(X_train)
        with timer(' GPU '):
            for i in range(repeat):
                centers, pred = fit(X_train, n_clusters, max_iter,
                                    use_custom_kernel)
        if output is not None:
            index = np.random.choice(10000000, 300, replace=False)
            draw(X_train[index], n_clusters, centers, pred[index], output)


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
