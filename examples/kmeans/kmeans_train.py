import argparse
import contextlib
import time

import numpy as np

import cupy
import kmeans


@contextlib.contextmanager
def timer(message):
    cupy.cuda.Stream.null.synchronize()
    start = time.time()
    yield
    cupy.cuda.Stream.null.synchronize()
    end = time.time()
    print('%s:  %f sec' % (message, end - start))


def run(gpuid, n_clusters, max_iter, elem, output):
    samples = np.random.randn(50000, 1000).astype(np.float32)
    X_train = np.r_[samples + 1, samples - 1]
    repeat = 1

    with timer(' CPU '):
        for i in range(repeat):
            centers, pred = kmeans.use_custom_kernel(X_train, n_clusters,
                                                     max_iter, elem)

    with cupy.cuda.Device(gpuid):
        X_train = cupy.asarray(X_train)
        with timer(' GPU '):
            for i in range(repeat):
                centers, pred = kmeans.use_custom_kernel(X_train, n_clusters,
                                                         max_iter, elem)
        if output is not None:
            kmeans.draw(X_train, n_clusters, centers, pred, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', '-g', default=0, type=int, dest='gpuid',
                        help='ID of GPU.')
    parser.add_argument('--n-clusters', '-n', default=2, type=int,
                        dest='n_clusters', help='number of clusters')
    parser.add_argument('--maxiter', '-m', default=10, type=int,
                        dest='max_iter', help='number of iterations')
    parser.add_argument('--elem', action='store_true', default=False,
                        help='use Elementwise kernel')
    parser.add_argument('--output', '-o', default=None, type=str,
                        dest='output', help='output image file name')
    args = parser.parse_args()
    run(args.gpuid, args.n_clusters, args.max_iter, args.elem, args.output)
