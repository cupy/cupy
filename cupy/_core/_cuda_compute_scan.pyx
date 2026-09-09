import threading
import warnings

from cupy._core._dtype cimport get_dtype
from cupy._core.core cimport _ndarray_base, compile_to_ltoir
from cupy._core cimport _scalar
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
            if hasattr(compute, 'OpKind'):
                _cuda_compute = compute
            else:
                warnings.warn(
                    'cuda.compute is installed but its CUDA bindings '
                    'could not be loaded, so the cuda_compute '
                    'accelerator will be skipped', RuntimeWarning)
                _cuda_compute = None
    return _cuda_compute


@_util.memoize(for_each_device=True)
def _make_raw_op(str src, str name):
    ltoir = compile_to_ltoir(src)
    return _get_cuda_compute().op.RawOp(ltoir=ltoir, name=name)


cdef str _get_generic_op(str op, dtype):
    type_decls = set()
    typename = _scalar.get_typename(dtype, type_decls)
    type_decls = _scalar.format_type_decls(type_decls)

    if op == 'PLUS':
        op = '+'
    elif op == 'MULTIPLIES':
        op = '*'
    else:
        raise ValueError(f'Unsupported operation: {op}')

    src = f'''
        #include <cupy/complex.cuh>
        {type_decls}
        using T = {typename};

        extern "C" __device__ void op(void* a, void* b, void* result) {{
            const T& pa = *static_cast<const T*>(a);
            const T& pb = *static_cast<const T*>(b);
            T& out = *static_cast<T*>(result);
            out = pa {op} pb;
        }}
    '''
    return src


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

    cc_dtype = compute.types.from_numpy_dtype(dtype)
    # If the dtype is not a storage type in cuda-compute, assume that
    # the OpKind will support +/* the same way that CuPy would.
    # (For more complex handling, OpKind should be attached to loops)
    if cc_dtype.info.typenum != cc_dtype.info.typenum.STORAGE:
        scan_op = getattr(compute.OpKind, op)
        op_src = ''
    else:
        op_src = _get_generic_op(op, dtype)
        scan_op = _make_raw_op(op_src, 'op')

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


cpdef cuda_compute_scan(_ndarray_base a, _ndarray_base result, dtype,
                        str op):
    """Perform an inclusive sum or product scan of `a` with cuda.compute.

    If the specified scan is not possible, None is returned.
    """
    if _get_cuda_compute() is None:
        return None

    # Note, we currently assume a simple +/* works for all dtypes
    # (and if it doesn't it fails the same for the default path).
    out_dtype = get_dtype(dtype) if result is None else result.dtype

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
