import threading
import warnings

from cupy._core._dtype cimport get_dtype
from cupy._core.core cimport _ndarray_base
from cupy.cuda.device cimport get_compute_capability

import numpy

import cupy
from cupy import _util
from cupy.cuda import compiler
from cupy.cuda._compiler_cache import _hash_hexdigest

_cuda_compute = False


cpdef _get_cuda_compute():
    global _cuda_compute

    if _cuda_compute is False:
        try:
            from cuda import compute
        except ImportError:
            _cuda_compute = None
        else:
            if getattr(compute, '_BINDINGS_AVAILABLE', True):
                _cuda_compute = compute
            else:
                warnings.warn(
                    'cuda.compute is installed but its CUDA bindings '
                    'could not be loaded, so the cuda_compute '
                    'accelerator will be skipped', RuntimeWarning)
                _cuda_compute = None
    return _cuda_compute


cdef _compile_cpp_to_ltoir(str src):
    return compiler._compile_module_with_cache(src, (), to_ltoir=True)


@_util.memoize(for_each_device=True)
def _make_raw_op(str src, str name):
    ltoir = _compile_cpp_to_ltoir(src)
    return _get_cuda_compute().op.RawOp(ltoir=ltoir, name=name)


cdef str _complex_op_src(str op, str ftype):
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
    return src % {'t': ftype, 'a': 8 if ftype == 'float' else 16}


cdef object _thread_local = threading.local()
_cache_key_prefix = None


cdef str _scanner_cache_name(str op, dtype, str op_src):
    global _cache_key_prefix
    if _cache_key_prefix is None:
        import cuda.cccl
        _cache_key_prefix = '|'.join((
            cuda.cccl.__version__,
            str(cupy.cuda.runtime.runtimeGetVersion()),
            compiler._get_cupy_cache_key()))
    cc = get_compute_capability()
    key_src = f'{_cache_key_prefix}|{cc}|{op}|{dtype.str}|{op_src}'.encode()
    return _hash_hexdigest(key_src) + '.cc_scan'


cdef _get_scanner(str op, dtype):
    cache = getattr(_thread_local, 'scanners', None)
    if cache is None:
        cache = _thread_local.scanners = {}
    key = (cupy.cuda.device.get_device_id(), op, dtype)
    cached = cache.get(key)
    if cached is not None:
        return cached

    compute = _get_cuda_compute()
    if dtype.kind == 'c':
        ftype = 'float' if dtype == numpy.dtype('complex64') else 'double'
        op_src = _complex_op_src(op, ftype)
        scan_op = _make_raw_op(op_src, 'op')
    else:
        op_src = ''
        scan_op = getattr(compute.OpKind, op)

    name = _scanner_cache_name(op, dtype, op_src)
    scanner = None
    blob = compiler._kernel_cache_backend.load(name)
    if blob is not None:
        try:
            scanner = compute.deserialize(blob)
        except Exception:
            pass
    if scanner is None:
        d = compute.ProxyArray(dtype)
        scanner = compute.make_inclusive_scan(d_in=d, d_out=d, op=scan_op)
        blob = scanner.serialize()
        compiler._kernel_cache_backend.save(name, blob, '')
    cache[key] = (scanner, scan_op)
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
