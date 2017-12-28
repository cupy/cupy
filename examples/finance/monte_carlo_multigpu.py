import argparse
import contextlib
import sys
import time

import cupy
import numpy

from black_scholes import black_scholes_kernel
from monte_carlo import monte_carlo_kernel

# CuPy also implements a feature to call kernels in different GPUs.
# Through this sample, we will explain how to allocate arrays
# in different devices, and call kernels in parallel.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='*',
                        default=[0], help='GPU IDs')
    parser.add_argument('--n-options', default=1000, type=int)
    parser.add_argument('--n-samples-per-thread', default=1000, type=int)
    parser.add_argument('--n-threads-per-option', default=10000, type=int)
    args = parser.parse_args()

    if len(args.gpus) == 0:
        print('At least one GPU is required.')
        sys.exit(1)

    def rand_range(m, M):
        samples = numpy.random.rand(args.n_options)
        return (m + (M - m) * samples).astype(numpy.float64)

    print('initializing...')
    stock_price_cpu = rand_range(5, 30)
    option_strike_cpu = rand_range(1, 100)
    option_years_cpu = rand_range(0.25, 10)
    risk_free = 0.02
    volatility = 0.3

    stock_price_gpus = []
    option_strike_gpus = []
    option_years_gpus = []
    call_prices_gpus = []
    print('start computation')
    print('    # of gpus: {}'.format(len(args.gpus)))
    print('    # of options: {}'.format(args.n_options))
    print('    # of samples per option: {}'.format(
        len(args.gpus) * args.n_samples_per_thread * args.n_threads_per_option)
    )
    # Allocate arrays in different devices
    for gpu_id in args.gpus:
        with cupy.cuda.Device(gpu_id):
            stock_price_gpus.append(cupy.array(stock_price_cpu))
            option_strike_gpus.append(cupy.array(option_strike_cpu))
            option_years_gpus.append(cupy.array(option_years_cpu))
            call_prices_gpus.append(cupy.empty(
                (args.n_options, args.n_threads_per_option),
                dtype=numpy.float64))

    @contextlib.contextmanager
    def timer(message):
        cupy.cuda.Stream.null.synchronize()
        start = time.time()
        yield
        cupy.cuda.Stream.null.synchronize()
        end = time.time()
        print('%s:\t%f sec' % (message, end - start))

    with timer('GPU (CuPy, Monte Carlo method)'):
        for i, gpu_id in enumerate(args.gpus):
            # Performs Monte-Carlo simulations in parallel
            with cupy.cuda.Device(gpu_id):
                monte_carlo_kernel(
                    stock_price_gpus[i][:, None],
                    option_strike_gpus[i][:, None],
                    option_years_gpus[i][:, None],
                    risk_free, volatility, args.n_samples_per_thread, i,
                    call_prices_gpus[i])

    # Transfer the result from the GPUs
    call_prices = [c.get() for c in call_prices_gpus]
    call_mc = numpy.concatenate(call_prices).reshape(
        len(args.gpus), args.n_options, args.n_threads_per_option)
    call_mc = call_mc.mean(axis=(0, 2))
    # Compute the error between the value of the exact solution
    # and that of the Monte-Carlo simulation
    with cupy.cuda.Device(args.gpus[0]):
        call_bs = black_scholes_kernel(
            stock_price_gpus[0], option_strike_gpus[0], option_years_gpus[0],
            risk_free, volatility)[0].get()
    error = cupy.std(call_mc - call_bs)
    print('Error: %f' % error)
