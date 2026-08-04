from cupy._core._dtype cimport get_dtype

import numpy

import cupy
from cupy import _util
from cupy.cuda import compiler

try:
    from cuda import compute as cuda_compute
except ImportError:
    cuda_compute = None


def _compile_cpp_to_ltoir(str src):
    return compiler._compile_module_with_cache(src, (), to_ltoir=True)


@_util.memoize(for_each_device=True)
def _make_raw_op(str src, str name):
    ltoir = _compile_cpp_to_ltoir(src)
    return cuda_compute.op.RawOp(ltoir=ltoir, name=name)


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


@_util.memoize(for_each_device=True)
def _get_scanner(int scan_op, dtype):
    if dtype.kind == 'c':
        ftype = 'float' if dtype == numpy.dtype('complex64') else 'double'
        if scan_op == CC_SCAN_SUM:
            op = _complex_plus_op(ftype)
        else:
            op = _complex_multiplies_op(ftype)
    else:
        if scan_op == CC_SCAN_SUM:
            op = cuda_compute.OpKind.PLUS
        else:
            op = cuda_compute.OpKind.MULTIPLIES
    d = cupy.empty(1, dtype=dtype)
    scanner = cuda_compute.make_inclusive_scan(d_in=d, d_out=d, op=op)
    return scanner, op


def _cc_scan_arrays(int scan_op, x, out):
    scanner, op = _get_scanner(scan_op, x.dtype)
    tmp_size = scanner(temp_storage=None, d_in=x, d_out=out,
                       num_items=x.size, op=op, init_value=None)

    d_tmp = cupy.empty(tmp_size, dtype=numpy.uint8)
    scanner(temp_storage=d_tmp, d_in=x, d_out=out, num_items=x.size,
            op=op, init_value=None,
            stream=cupy.cuda.get_current_stream())


cpdef bint _supports_dtype(dtype) except *:
    if cuda_compute is None:
        return False

    # supports signed/unsigned int, floating, complex dtypes
    if dtype.kind not in 'iufc':
        return False

    return True


cpdef cc_scan(a, result, dtype, int scan_op):
    """Perform an inclusive sum or product scan of `a` with cuda.compute.

     If the specified scan is not possible, None is returned.
    """

    dtype = get_dtype(dtype)

    if result is not None:
        if not _supports_dtype(result.dtype):
            return None
        _cc_scan_arrays(scan_op, result, result)
        return result

    if not _supports_dtype(dtype):
        return None

    # avoid unneccessary astype() copy when promotion is not needed
    # cc assumes contigous input, would misread strided inputs
    if a.dtype == dtype and a.flags.c_contiguous:
        result = cupy.empty(a.size, dtype=dtype)
        _cc_scan_arrays(scan_op, a.ravel(), result)
        return result

    result = a.astype(dtype, order='C').ravel()
    _cc_scan_arrays(scan_op, result, result)
    return result
