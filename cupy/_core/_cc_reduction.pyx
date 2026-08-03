from cupy._core._carray cimport shape_t
from cupy._core.core cimport _ndarray_base

import numpy

import cupy
from cupy import _util
from cupy.cuda import device

try:
    from cuda import compute as cuda_compute
    from cuda.core import Program, ProgramOptions
except ImportError:
    cuda_compute = None

cdef object _float16 = numpy.dtype('float16')


def _compile_cpp_to_ltoir(str src):
    options = ProgramOptions(
        arch='sm_' + device.get_compute_capability(),
        relocatable_device_code=True,
        link_time_optimization=True,
    )
    return Program(src, 'c++', options=options).compile('ltoir').code


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


@_util.memoize(for_each_device=True)
def _get_sum_reducer(in_dtype, out_dtype):
    d_in = cupy.empty(1, dtype=in_dtype)
    d_out = cupy.empty((), dtype=out_dtype)
    init = numpy.zeros((), dtype=out_dtype)
    if in_dtype.kind == 'c':
        ftype = 'float' if in_dtype == numpy.dtype('complex64') else 'double'
        op = _complex_plus_op(ftype)
    else:
        op = cuda_compute.OpKind.PLUS
    red = cuda_compute.make_reduce_into(
        d_in=d_in, d_out=d_out, h_init=init, op=op)
    return red, op, init


def _cc_device_sum(x, out):
    size = x.size
    red, op, init = _get_sum_reducer(x.dtype, out.dtype)

    tmp_size = red(temp_storage=None, d_in=x, d_out=out,
                   num_items=size, op=op, h_init=init)
    d_tmp = cupy.empty(tmp_size, dtype=numpy.uint8)
    red(temp_storage=d_tmp, d_in=x, d_out=out, num_items=size,
        op=op, h_init=init, stream=cupy.cuda.get_current_stream())


cpdef bint _can_use_cc_reduction(
        str name, list in_args, list out_args, tuple out_axis) except *:
    cdef _ndarray_base x, ret

    if cuda_compute is None:
        return False

    # TODO: extend to other reduction ops
    if name not in ('cupy_sum', 'cupy_sum_with_dtype'):
        return False

    # reduce to a scalar
    if len(out_axis) != 0:
        return False

    # support reductions with only 1 input and 1 output
    if len(in_args) != 1 or len(out_args) != 1:
        return False

    if not isinstance(in_args[0], _ndarray_base):
        return False

    x = in_args[0]
    ret = out_args[0]

    # TODO: add support for fp16
    if x.dtype == _float16 or ret.dtype == _float16:
        return False

    # TODO: support cp.sum(float_arr, dtype=cp.complex64)
    if (x.dtype.kind == 'c') != (ret.dtype.kind == 'c'):
        return False

    if not x._c_contiguous:
        return False

    return True


cdef bint _try_to_call_cc_reduction(
        self, list in_args, list out_args, const shape_t& a_shape,
        stream, tuple reduce_axis, tuple out_axis, const shape_t& out_shape,
        _ndarray_base ret) except *:
    """Try to use cuda.compute (CUB DeviceReduce).

    Updates `ret` and returns a boolean value whether cuda.compute is used.

    Note: only currently supports plain sum, no fp16, no mixed
    real/complex dtypes, no axis (segmented) reduction, and no strided
    input; unsupported cases fall back to the CUB / generic path.
    """
    if not _can_use_cc_reduction(self.name, in_args, out_args, out_axis):
        return False

    _cc_device_sum(in_args[0], ret)
    return True
