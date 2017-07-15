import argparse
import contextlib
import time

import matplotlib
matplotlib.use('Agg')
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


def fit(X, n_clusters, max_iter, use_custom_kernel):
    assert X.ndim == 2
    xp = cupy.get_array_module(X)
    pred = xp.zeros(len(X), dtype=np.int32)
    initial_indexes = np.random.choice(len(X), n_clusters,
                                       replace=False).astype(np.int32)
    centers = X[initial_indexes]
    data_num = X.shape[0]
    data_dim = X.shape[1]

    for _ in six.moves.range(max_iter):
        # calculate distances and label
        if not use_custom_kernel or xp == np:
            distances = xp.linalg.norm(X[:, None, :] - centers[None, :, :],
                                       axis=2)
        else:
            distances = xp.zeros((data_num, n_clusters), dtype=np.float32)
            cupy.ElementwiseKernel(
                'S data, raw S centers, int32 n_clusters, int32 dim',
                'raw S dist',
                '''
                for (int j = 0; j < n_clusters; j++){
                    int cent_ind[] = {j, i % dim};
                    int dist_ind[] = {i / dim, j};
                    double diff = centers[cent_ind] - data;
                    atomicAdd(&dist[dist_ind], diff * diff);
                }
                ''',
                'calc_distances'
            )(X, centers, n_clusters, data_dim, distances)

        new_pred = xp.argmin(distances, axis=1).astype(np.int32)
        if xp.all(new_pred == pred):
            break
        pred = new_pred

        # calculate centers
        if not use_custom_kernel or xp == np:
            centers = xp.stack([X[pred == i].mean(axis=0)
                                for i in six.moves.range(n_clusters)])
        else:
            centers = xp.zeros((n_clusters, data_dim),
                               dtype=np.float32)
            group = xp.zeros(n_clusters, dtype=np.float32)
            label = pred[:, None]
            cupy.ElementwiseKernel(
                'S data, T label, int32 dim', 'raw S centers, raw S group',
                '''
                int cent_ind[] = {label, i % dim};
                atomicAdd(&centers[cent_ind], data);
                atomicAdd(&group[label], 1);
                ''',
                'calc_center'
            )(X, label, data_dim, centers, group)
            group /= data_dim
            centers /= group[:, None]

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
