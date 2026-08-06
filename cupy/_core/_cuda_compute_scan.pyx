from cupy._core._cuda_compute_ops cimport (
    CUDA_COMPUTE_PLUS, CUDA_COMPUTE_MULTIPLIES)
from cupy._core._dtype cimport get_dtype
from cupy._core.core cimport _ndarray_base

import numpy

import cupy
from cupy import _util
from cupy.cuda import compiler


# lazy import: avoid importing cuda.compute whenever cupy is imported
@_util.memoize()
def _get_cuda_compute():
    try:
        from cuda import compute
    except ImportError:
        return None
    return compute


def _compile_cpp_to_ltoir(str src):
    return compiler._compile_module_with_cache(src, (), to_ltoir=True)


@_util.memoize(for_each_device=True)
def _make_raw_op(str src, str name):
    ltoir = _compile_cpp_to_ltoir(src)
    return _get_cuda_compute().op.RawOp(ltoir=ltoir, name=name)


def _complex_plus_op(str ftype):
    src = '''
    struct cplx { %(t)s re, im; };
    extern "C" __device__ void op(void* a, void* b, void* result) {
        const cplx* pa = static_cast<const cplx*>(a);
        const cplx* pb = static_cast<const cplx*>(b);
        static_cast<cplx*>(result)->re = pa->re + pb->re;
        static_cast<cplx*>(result)->im = pa->im + pb->im;
    }
    ''' % {'t': ftype}
    return _make_raw_op(src, 'op')


def _complex_multiplies_op(str ftype):
    src = '''
    struct cplx { %(t)s re, im; };
    extern "C" __device__ void op(void* a, void* b, void* result) {
        const cplx* pa = static_cast<const cplx*>(a);
        const cplx* pb = static_cast<const cplx*>(b);
        %(t)s re = pa->re * pb->re - pa->im * pb->im;
        %(t)s im = pa->re * pb->im + pa->im * pb->re;
        static_cast<cplx*>(result)->re = re;
        static_cast<cplx*>(result)->im = im;
    }
    ''' % {'t': ftype}
    return _make_raw_op(src, 'op')


def _get_scanner(int op, dtype):
    compute = _get_cuda_compute()
    if dtype.kind == 'c':
        ftype = 'float' if dtype == numpy.dtype('complex64') else 'double'
        if op == CUDA_COMPUTE_PLUS:
            scan_op = _complex_plus_op(ftype)
        elif op == CUDA_COMPUTE_MULTIPLIES:
            scan_op = _complex_multiplies_op(ftype)
        else:
            raise ValueError(f'unsupported scan op: {op}')
    else:
        if op == CUDA_COMPUTE_PLUS:
            scan_op = compute.OpKind.PLUS
        elif op == CUDA_COMPUTE_MULTIPLIES:
            scan_op = compute.OpKind.MULTIPLIES
        else:
            raise ValueError(f'unsupported scan op: {op}')
    d = compute.ProxyArray(dtype)
    scanner = compute.make_inclusive_scan(d_in=d, d_out=d, op=scan_op)
    return scanner, scan_op


def _cuda_compute_scan_arrays(int op, x, out):
    scanner, scan_op = _get_scanner(op, x.dtype)
    tmp_size = scanner(temp_storage=None, d_in=x, d_out=out,
                       num_items=x.size, op=scan_op, init_value=None)

    d_tmp = cupy.empty(tmp_size, dtype=numpy.uint8)
    scanner(temp_storage=d_tmp, d_in=x, d_out=out, num_items=x.size,
            op=scan_op, init_value=None,
            stream=cupy.cuda.get_current_stream())


cpdef bint _supports_dtype(dtype) except *:
    # supports signed/unsigned int, floating, complex dtypes
    if dtype.kind not in 'iufc':
        return False

    return True


cpdef cuda_compute_scan(_ndarray_base a, _ndarray_base result, dtype, int op):
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
