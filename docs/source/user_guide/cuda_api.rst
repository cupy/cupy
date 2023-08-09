Accessing CUDA Functionalities
==============================

.. _cuda_stream_event:

Streams and Events
------------------

In this section we discuss basic usages for CUDA streams and events. For the API reference please see
:ref:`stream_event_api`. For their roles in the CUDA programming model, please refer to `CUDA Programming Guide`_.

CuPy provides high-level Python APIs :class:`~cupy.cuda.Stream` and :class:`~cupy.cuda.Event` for creating
streams and events, respectively. Data copies and kernel launches are enqueued onto the :ref:`current_stream`,
which can be queried via :func:`~cupy.cuda.get_current_stream` and changed either by setting up a context
manager:

.. doctest::

    >>> import numpy as np
    >>>
    >>> a_np = np.arange(10)
    >>> s = cp.cuda.Stream()
    >>> with s:
    ...     a_cp = cp.asarray(a_np)  # H2D transfer on stream s
    ...     b_cp = cp.sum(a_cp)      # kernel launched on stream s
    ...     assert s == cp.cuda.get_current_stream()
    ...
    >>> # fall back to the previous stream in use (here the default stream)
    >>> # when going out of the scope of s

or by using the :meth:`~cupy.cuda.Stream.use` method:

.. doctest::

    >>> s = cp.cuda.Stream()
    >>> s.use()  # any subsequent operations are done on steam s  # doctest: +ELLIPSIS
    <Stream ... (device ...)>
    >>> b_np = cp.asnumpy(b_cp)
    >>> assert s == cp.cuda.get_current_stream()
    >>> cp.cuda.Stream.null.use()  # fall back to the default (null) stream
    <Stream 0 (device -1)>
    >>> assert cp.cuda.Stream.null == cp.cuda.get_current_stream()

Events can be created either manually or through the :meth:`~cupy.cuda.Stream.record` method.
:class:`~cupy.cuda.Event` objects can be used for timing GPU activities (via :func:`~cupy.cuda.get_elapsed_time`)
or setting up inter-stream dependencies:

.. doctest::

    >>> e1 = cp.cuda.Event()
    >>> e1.record()
    >>> a_cp = b_cp * a_cp + 8
    >>> e2 = cp.cuda.get_current_stream().record()
    >>>
    >>> # set up a stream order
    >>> s2 = cp.cuda.Stream()
    >>> s2.wait_event(e2)
    >>> with s2:
    ...     # the a_cp is guaranteed updated when this copy (on s2) starts
    ...     a_np = cp.asnumpy(a_cp)
    >>>
    >>> # timing
    >>> e2.synchronize()
    >>> t = cp.cuda.get_elapsed_time(e1, e2)  # only include the compute time, not the copy time

Just like the :class:`~cupy.cuda.Device` objects, :class:`~cupy.cuda.Stream` and :class:`~cupy.cuda.Event`
objects can also be used for synchronization.

.. note::

    In CuPy, the :class:`~cupy.cuda.Stream` objects are managed on the per thread, per device basis.

.. note::

    On NVIDIA GPUs, there are two stream singleton objects :obj:`~cupy.cuda.Stream.null` and
    :obj:`~cupy.cuda.Stream.ptds`, referred to as the *legacy* default stream and the *per-thread* default
    stream, respectively. CuPy uses the former as default when no user-defined stream is in use. To
    change this behavior, set the environment variable ``CUPY_CUDA_PER_THREAD_DEFAULT_STREAM`` to 1,
    see :ref:`environment`. This is not applicable to AMD GPUs.

.. _CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

To interoperate with streams created in other Python libraries, CuPy provides the :class:`~cupy.cuda.ExternalStream`
API to wrap an existing stream pointer (given as a Python ``int``). See :doc:`interoperability` for details.

CUDA Driver and Runtime API
---------------------------

Under construction. Please see :ref:`runtime_api` for the API reference.
