import threading
from importlib import metadata

from cupy._core._dtype cimport get_dtype
from cupy._core.core cimport _ndarray_base
from cupy.cuda.device cimport get_compute_capability
from cupy._core._cuda_compute_common cimport _get_cuda_compute

import numpy

import cupy
from cupy._core._cuda_compute_common import _make_raw_op
from cupy.cuda import compiler
from cupy.cuda._compiler_cache import _hash_hexdigest

cdef dict _complex_op_srcs = {}


cdef _complex_op(str op, str ftype):
    src = _complex_op_srcs.get((op, ftype))
    if src is None:
        if op == 'PLUS':
            body = '''
        static_cast<cplx*>(result)->re = pa->re + pb->re;
        static_cast<cplx*>(result)->im = pa->im + pb->im;'''
        else:
            body = '''
        %(t)s re = pa->re * pb->re - pa->im * pb->im;
        %(t)s im = pa->re * pb->im + pa->im * pb->re;
        static_cast<cplx*>(result)->re = re;
        static_cast<cplx*>(result)->im = im;'''
        src = '''
    struct __align__(%(a)d) cplx { %(t)s re, im; };
    extern "C" __device__ void op(void* a, void* b, void* result) {
        const cplx* pa = static_cast<const cplx*>(a);
        const cplx* pb = static_cast<const cplx*>(b);''' + body + '''
    }
    '''
        src = src % {'t': ftype, 'a': 8 if ftype == 'float' else 16}
        _complex_op_srcs[(op, ftype)] = src
    return _make_raw_op(src, 'op')


cdef object _thread_local = threading.local()
_cache_key_prefix = None


cdef str _scanner_cache_name(str op, dtype):
    global _cache_key_prefix
    if _cache_key_prefix is None:
        _cache_key_prefix = '|'.join((
            metadata.version('cuda-cccl'),
            str(cupy.cuda.runtime.runtimeGetVersion())))
    cc = get_compute_capability()
    key_src = f'{_cache_key_prefix}|{cc}|{op}|{dtype.str}'.encode()
    return _hash_hexdigest(key_src) + '.cc_scan'


cdef _get_scanner(str op, dtype):
    compute = _get_cuda_compute()
    if dtype.kind == 'c':
        ftype = 'float' if dtype == numpy.dtype('complex64') else 'double'
        scan_op = _complex_op(op, ftype)
    else:
        scan_op = getattr(compute.OpKind, op)

    cache = getattr(_thread_local, 'scanners', None)
    if cache is None:
        cache = _thread_local.scanners = {}
    key = (cupy.cuda.device.get_device_id(), op, dtype.char)
    scanner = cache.get(key)
    if scanner is not None:
        return scanner, scan_op

    name = _scanner_cache_name(op, dtype)
    blob = compiler._kernel_cache_backend.load(name)
    if blob is not None:
        scanner = compute.deserialize(blob)
    else:
        d = compute.ProxyArray(dtype)
        scanner = compute.make_inclusive_scan(d_in=d, d_out=d, op=scan_op)
        blob = scanner.serialize()
        compiler._kernel_cache_backend.save(name, blob, '')
    cache[key] = scanner
    return scanner, scan_op


def _cuda_compute_scan_arrays(str op, x, out):
    scanner, scan_op = _get_scanner(op, x.dtype)
    tmp_size = scanner(temp_storage=None, d_in=x, d_out=out,
                       num_items=x.size, op=scan_op, init_value=None)

    d_tmp = cupy.empty(tmp_size, dtype=numpy.uint8)
    scanner(temp_storage=d_tmp, d_in=x, d_out=out, num_items=x.size,
            op=scan_op, init_value=None,
            stream=cupy.cuda.get_current_stream())


cpdef bint _supports_dtype(dtype) except ?-1:
    if dtype.kind not in 'iufc':
        return False

    return True


cpdef cuda_compute_scan(_ndarray_base a, _ndarray_base result, dtype,
                        str op):
    """Perform an inclusive sum or product scan of `a` with cuda.compute.

    If the specified scan is not possible, None is returned.
    """
    if _get_cuda_compute() is None:
        return None

    out_dtype = get_dtype(dtype) if result is None else result.dtype
    if not _supports_dtype(out_dtype):
        return None

    if result is None:
        if a.dtype == out_dtype and a._c_contiguous:
            # No promotion needed: scan straight out of `a`, skipping the
            # astype copy. c_contiguous keeps a.ravel() a free view (a
            # strided `a` would copy) and guarantees cuda.compute reads
            # the elements linearly.
            src, result = a.ravel(), cupy.empty(a.size, dtype=out_dtype)
        else:
            # Promotion and/or strided input: astype(order='C') converts
            # and contiguizes, making the ravel() free. Scan is in-place.
            src = result = a.astype(out_dtype, order='C').ravel()
    else:
        src = result  # scan_core already copied `a` into result

    _cuda_compute_scan_arrays(op, src, result)
    return result
