Performance Best Practices
==========================

Here we gather a few tricks and advices for improving CuPy's performance.

Benchmarking
------------

It is utterly important to first identify the performance bottleneck before making any attempt to optimize
your code. To help set up a baseline benchmark, CuPy provides a useful utility :func:`cupyx.profiler.repeat`
for timing the elapsed time of a Python function on both CPU and GPU:

.. doctest::

    >>> from cupyx.profiler import repeat
    >>> 
    >>> def my_func(a):
    ...     return cp.sqrt(cp.sum(a**2, axis=-1))
    ... 
    >>> a = cp.random.random((256, 1024))
    >>> print(repeat(my_func, (a,), n_repeat=20))  # doctest: +SKIP
    my_func             :    CPU:   44.407 us   +/- 2.428 (min:   42.516 / max:   53.098) us     GPU-0:  181.565 us   +/- 1.853 (min:  180.288 / max:  188.608) us

Because GPU executions run asynchronously with respect to CPU executions, a common pitfall in GPU programming is to mistakenly
measure the elapsed time using CPU timing utilities (such as :py:func:`time.perf_counter` from the Python Standard Library
or the ``%timeit`` magic from IPython), which have no knowledge in the GPU runtime. :func:`cupyx.profiler.repeat` addresses
this by setting up CUDA events on the :ref:`current_stream` right before and after the function to be measured and
synchronizing over the end event (see :ref:`cuda_stream_event` for detail). Below we sketch what is done internally in :func:`cupyx.profiler.repeat`:

.. doctest::

    >>> import time
    >>> start_gpu = cp.cuda.Event()
    >>> end_gpu = cp.cuda.Event()
    >>>
    >>> start_gpu.record()
    >>> start_cpu = time.perf_counter()
    >>> out = my_func(a)
    >>> end_cpu = time.perf_counter()
    >>> end_gpu.record()
    >>> end_gpu.synchronize()
    >>> t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    >>> t_cpu = end_cpu - start_cpu

Additionally, :func:`cupyx.profiler.repeat` runs a few warm-up runs to reduce timing fluctuation and exclude the overhead in first invocations.


In-depth profiling
------------------

Under construction.


Use CUB/cuTENSOR backends for reduction operations
--------------------------------------------------

Under construction.


Overlapping work using streams
------------------------------

Under construction.


Use JIT compiler
----------------

Under construction. For now please refer to :ref:`jit_kernel_definition` for a quick introduction.


Prefer float32 over float64
---------------------------

Under construction.
