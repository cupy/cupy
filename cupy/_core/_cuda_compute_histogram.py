from __future__ import annotations

import threading

import numpy

import cupy
from cupy._core import _cuda_compute_common
from cupy.cuda import compiler
from cupy.cuda import memory
from cupy.cuda._compiler_cache import _hash_hexdigest
from cupy.cuda.device import get_compute_capability


_thread_local = threading.local()


def _histogram_cache_name(dtype, n_bins):
    prefix = _cuda_compute_common._environment_cache_key_prefix()
    cc = get_compute_capability()
    key_src = f'{prefix}|{cc}|{dtype.str}|{n_bins}'.encode()
    return _hash_hexdigest(key_src) + '.cc_histogram'


def _get_histogram(dtype, n_bins):
    cache = getattr(_thread_local, 'histograms', None)
    if cache is None:
        cache = _thread_local.histograms = {}

    key = (cupy.cuda.device.get_device_id(), dtype, n_bins)
    cached = cache.get(key)
    if cached is not None:
        return cached

    compute = _cuda_compute_common._get_cuda_compute()
    name = _histogram_cache_name(dtype, n_bins)
    histogram = None
    blob = compiler._kernel_cache_backend.load(name)
    if blob is not None:
        try:
            histogram = compute.deserialize(blob)
        except Exception:
            pass
    if histogram is None:
        histogram = compute.make_histogram_even(
            d_samples=compute.ProxyArray(dtype),
            d_histogram=compute.ProxyArray(numpy.dtype(numpy.uint64)),
            h_num_output_levels=numpy.array([n_bins + 1], dtype=numpy.int32),
            h_lower_level=numpy.array([0], dtype=numpy.int64),
            h_upper_level=numpy.array([n_bins], dtype=numpy.int64),
            num_samples=0)
        blob = histogram.serialize()
        compiler._kernel_cache_backend.save(name, blob, '')
    cache[key] = histogram
    return histogram


def cuda_compute_bincount(x, b, n_bins):
    """Count occurrences of each value in `x` into `b` with cuda.compute.

    If the bincount is not possible with cuda.compute, None is returned.
    """
    if _cuda_compute_common._get_cuda_compute() is None:
        return None
    if x.dtype.kind not in 'bui':
        return None
    # num_samples and h_num_output_levels are int32 in histogram_even
    if x.size > 0x7fffffff or n_bins + 1 > 0x7fffffff:
        return None

    x = cupy.ascontiguousarray(x)
    histogram = _get_histogram(x.dtype, n_bins)
    # histogram counters must be uint64 (there is no signed 64-bit
    # atomicAdd), so write into b through a uint64 view; every count
    # is at most x.size < 2**31, small enough that the same bits mean
    # the same number in both types
    counts = b.view(numpy.uint64)
    num_output_levels = numpy.array([n_bins + 1], dtype=numpy.int32)
    lower_level = numpy.array([0], dtype=numpy.int64)
    upper_level = numpy.array([n_bins], dtype=numpy.int64)
    tmp_size = histogram(
        temp_storage=None, d_samples=x, d_histogram=counts,
        h_num_output_levels=num_output_levels, h_lower_level=lower_level,
        h_upper_level=upper_level, num_samples=x.size)
    try:
        d_tmp = cupy.empty(tmp_size, dtype=numpy.uint8)
    except memory.OutOfMemoryError:
        return None
    histogram(
        temp_storage=d_tmp, d_samples=x, d_histogram=counts,
        h_num_output_levels=num_output_levels, h_lower_level=lower_level,
        h_upper_level=upper_level, num_samples=x.size,
        stream=cupy.cuda.get_current_stream())
    return b
