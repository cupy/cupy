import argparse
import contextlib
import time

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


def fit(A, b, tol, max_iter):
    # Note that this function works even tensors 'A' and 'b' are NumPy or CuPy
    # arrays.
    xp = cupy.get_array_module(A)
    x = xp.zeros_like(b, dtype=np.float64)
    r0 = b - xp.dot(A, x)
    p = r0
    for i in six.moves.range(max_iter):
        a = xp.inner(r0, r0) / xp.inner(p, xp.dot(A, p))
        x += a * p
        r1 = r0 - a * xp.dot(A, p)
        if xp.linalg.norm(r1) < tol:
            return x
        b = xp.inner(r1, r1) / xp.inner(r0, r0)
        p = r1 + b * p
        r0 = r1
    print('Failed to converge. Increase max-iter or tol.')
    return x


def run(gpu_id, tol, max_iter):
    """CuPy Conjugate gradient example

    Solve simultaneous linear equations, Ax = b.
    'A' and 'x' are created randomly and 'b' is computed by 'Ax' at first.
    Then, 'x' is computed from 'A' and 'b' in two ways, namely with CPU and
    GPU. To evaluate the accuracy of computation, the Euclidean distances
    between the answer 'x' and the reconstructed 'x' are computed.

    """
    for repeat in range(3):
        print('Trial: %d' % repeat)
        # Create the large symmetric matrix 'A'.
        N = 2000
        A = np.random.randint(-50, 50, size=(N, N))
        A = (A + A.T).astype(np.float64)
        x_ans = np.random.randint(-50, 50, size=N).astype(np.float64)
        b = np.dot(A, x_ans)

        print('Running CPU...')
        with timer(' CPU '):
            x_cpu = fit(A, b, tol, max_iter)
        print(np.linalg.norm(x_cpu - x_ans))

        with cupy.cuda.Device(gpu_id):
            A_gpu = cupy.asarray(A)
            b_gpu = cupy.asarray(b)
            print('Running GPU...')
            with timer(' GPU '):
                x_gpu = fit(A_gpu, b_gpu, tol, max_iter)
            print(np.linalg.norm(cupy.asnumpy(x_gpu) - x_ans))

        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', '-g', default=0, type=int,
                        help='ID of GPU.')
    parser.add_argument('--tol', '-t', default=0.1, type=float,
                        help='tolerance to stop iteration')
    parser.add_argument('--max-iter', '-m', default=5000, type=int,
                        help='number of iterations')
    args = parser.parse_args()
    run(args.gpu_id, args.tol, args.max_iter)
