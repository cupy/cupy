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


def run_kmeans(X_train, estimator):
    estimator.fit(X_train)


def run(gpuid, n_clusters, max_iter, tol, output):
    X_train = np.random.rand(1000, 50000) * 10
    repeat = 1

    estimator_cpu = kmeans.KMeans(n_clusters=n_clusters,
                                  max_iter=max_iter, tol=tol)
    with timer(' CPU '):
        for i in range(repeat):
            run_kmeans(X_train, estimator_cpu)
    if output is not None:
        estimator_cpu.draw(X_train, output)
 
    cupy.cuda.Device(gpuid)
    X_train = cupy.asarray(X_train)
    estimator_gpu = kmeans.KMeans(n_clusters=n_clusters,
                                  max_iter=max_iter, tol=tol)
    with timer(' GPU '):
        for i in range(repeat):
            run_kmeans(X_train, estimator_gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', '-g', default=0, type=int, dest='gpuid',
                        help='ID of GPU.')
    parser.add_argument('--n_clusters', '-n', default=2, type=int, dest='n_clusters',
                        help='')
    parser.add_argument('--max_iter', '-m', default=10, type=int, dest='max_iter',
                        help='')
    parser.add_argument('--tol', '-t', default=1e-4, type=float, dest='tol',
                        help='')
    parser.add_argument('--output', '-o', default=None, type=str, dest='output',
                        help='')
    args = parser.parse_args()
    run(args.gpuid, args.n_clusters, args.max_iter, args.tol, args.output)
