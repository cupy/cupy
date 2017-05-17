import argparse
import contextlib
import time

import numpy as np

import cupy
import gmm


@contextlib.contextmanager
def timer(message):
    cupy.cuda.Stream.null.synchronize()
    start = time.time()
    yield
    cupy.cuda.Stream.null.synchronize()
    end = time.time()
    print('%s:  %f sec' % (message, end - start))


def run_gmm(X_train, y_train, X_test, y_test, max_iter, tol):
    xp = cupy.get_array_module(X_train)
    inv_cov, means, weights, cov = gmm.train_gmm(X_train, max_iter, tol)
    y_train_pred = gmm.predict(X_train, inv_cov, means, weights)
    train_accuracy = xp.mean(y_train_pred == y_train) * 100
    y_test_pred = gmm.predict(X_test, inv_cov, means, weights)
    test_accuracy = xp.mean(y_test_pred == y_test) * 100
    print('train_accuracy : %f' % train_accuracy)
    print('test_accuracy : %f' % test_accuracy)
    return y_test_pred, means, cov


def run(gpuid, max_iter, tol, output):
    train1 = np.random.normal(3, [1, 2], size=(500000, 2)).astype(np.float32)
    train2 = np.random.normal(-3, [2, 1], size=(500000, 2)).astype(np.float32)
    X_train = np.r_[train1, train2]
    test1 = np.random.normal(3, [1, 2], size=(100, 2)).astype(np.float32)
    test2 = np.random.normal(-3, [2, 1], size=(100, 2)).astype(np.float32)
    X_test = np.r_[test1, test2]
    y_train = np.r_[np.zeros(500000), np.ones(500000)].astype(np.int32)
    y_test = np.r_[np.zeros(100), np.ones(100)].astype(np.int32)
    repeat = 1

    with timer(' CPU '):
        for i in range(repeat):
            y_test_pred, means, cov = run_gmm(X_train, y_train, X_test, y_test,
                                              max_iter, tol)

    with cupy.cuda.Device(gpuid):
        X_train_gpu = cupy.array(X_train)
        y_train_gpu = cupy.array(y_train)
        y_test_gpu = cupy.array(y_test)
        X_test_gpu = cupy.array(X_test)
        with timer(' GPU '):
            for i in range(repeat):
                y_test_pred, means, cov = run_gmm(X_train_gpu, y_train_gpu,
                                                  X_test_gpu, y_test_gpu,
                                                  max_iter, tol)
        if output is not None:
            gmm.draw(X_test_gpu, y_test_pred, means, cov, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', '-g', default=0, type=int, dest='gpuid',
                        help='ID of GPU.')
    parser.add_argument('--max-iter', '-m', default=30, type=int,
                        dest='max_iter', help='number of iterations')
    parser.add_argument('--tol', '-t', default=1e-3, type=float,
                        dest='tol', help='')
    parser.add_argument('--output-image', '-o', default=None, type=str,
                        dest='output', help='output image file name')
    args = parser.parse_args()
    run(args.gpuid, args.max_iter, args.tol, args.output)
