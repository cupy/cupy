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


def run_gmm(X_train, y_train, X_test, y_test, estimator, n_classes):
    xp = cupy.get_array_module(X_train)
    estimator.fit(X_train)
    y_train_pred = estimator.predict(X_train)
    train_accuracy = xp.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    y_test_pred = estimator.predict(X_test)
    test_accuracy = xp.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    return train_accuracy, test_accuracy


def run(gpuid, n_classes, max_iter):
    trains = np.random.rand(100, 2).astype(np.float32)  # change
    X_train = np.r_[trains + 0.3, trains - 0.3]
    tests = np.random.rand(10, 2).astype(np.float32)
    X_test = np.r_[tests + 0.3, tests - 0.3]
    y_train = np.r_[np.ones(100), np.zeros(100)].astype(np.int32)
    y_test = np.r_[np.ones(10), np.zeros(10)].astype(np.int32)
    repeat = 1

    estimator_cpu = gmm.GaussianMixture(n_components=n_classes,
                                        max_iter=max_iter)
    with timer(' CPU '):
        for i in range(repeat):
            train_acc, test_acc = run_gmm(X_train, y_train, X_test, y_test,
                                          estimator_cpu, n_classes)
    print('train_accuracy : %f' % train_acc)
    print('test_accuracy : %f' % test_acc)
    estimator_cpu.draw(X_train)

    with cupy.cuda.Device(gpuid):
        X_train_gpu = cupy.array(X_train)
        y_train_gpu = cupy.array(y_train)
        y_test_gpu = cupy.array(y_test)
        X_test_gpu = cupy.array(X_test)
        estimator_gpu = gmm.GaussianMixture(n_components=n_classes,
                                            max_iter=max_iter)
        with timer(' GPU '):
            for i in range(repeat):
                train_acc, test_acc = run_gmm(X_train_gpu, y_train_gpu,
                                              X_test_gpu, y_test_gpu,
                                              estimator_gpu, n_classes)
        print('train_accuracy : %f' % train_acc)
        print('test_accuracy : %f' % test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', '-g', default=0, type=int, dest='gpuid',
                        help='ID of GPU.')
    parser.add_argument('--n_classes', '-n', default=2, type=int,
                        dest='n_classes', help='number of classes')
    parser.add_argument('--max_iter', '-m', default=30, type=int,
                        dest='max_iter', help='number of iterations')
    args = parser.parse_args()
    run(args.gpuid, args.n_classes, args.max_iter)
