from __future__ import annotations

import numpy

import cupy
from cupy._core._cuda_compute_common import _get_cuda_compute
from cupy._core._cuda_compute_common import cached_algorithm
from cupy.cuda import memory


def _get_histogram(dtype, n_bins):
    # TODO: key on the kernel variant instead of n_bins once cuda.compute's
    # v2 backend takes the bin count at run time (NVIDIA/cccl#11418)
    compute = _get_cuda_compute()
    key = (dtype, n_bins)
    return cached_algorithm(
        'histogram', key, repr(key),
        lambda: compute.make_histogram_even(
            d_samples=compute.ProxyArray(dtype),
            d_histogram=compute.ProxyArray(numpy.dtype(numpy.uint64)),
            h_num_output_levels=numpy.array([n_bins + 1], dtype=numpy.int32),
            h_lower_level=numpy.array([0], dtype=numpy.int64),
            h_upper_level=numpy.array([n_bins], dtype=numpy.int64),
            num_samples=0))


def cuda_compute_bincount(x, b, n_bins):
    """Count occurrences of each value in `x` into `b` with cuda.compute.

    If the bincount is not possible with cuda.compute, None is returned.
    """
    if _get_cuda_compute() is None:
        return None
    if x.dtype.kind not in 'bui':
        return None
    # TODO: drop the x.size limit once cuda.compute's v2 backend takes
    # num_samples at run time
    if x.size > 0x7fffffff or n_bins + 1 > 0x7fffffff:
        return None

    # TODO: pass the strided view once cuda.compute exposes a 1-D strided
    # iterator (NVIDIA/cccl#11417)
    x = cupy.ascontiguousarray(x)
    histogram = _get_histogram(x.dtype, n_bins)
    # histogram counters must be uint64 (there is no signed 64-bit
    # atomicAdd), so write into b through a uint64 view
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
