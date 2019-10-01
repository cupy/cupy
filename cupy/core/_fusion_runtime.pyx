import string

from libcpp cimport vector
import numpy

from cupy.core import _scalar
from cupy.core import _dtype
from cupy.core import core
from cupy.core.core cimport compile_with_cache
from cupy.core.core cimport ndarray, Indexer
from cupy.core cimport internal
from cupy.core import _fusion_op
from cupy.core cimport _routines_manipulation as _manipulation
from cupy import util
from cupy.core._fusion_shape import _AbstractDim
from cupy.core._fusion_variable import _FusionCudaVarBase
from cupy.core._fusion_variable import _FusionCudaScalar
from cupy.core._fusion_variable import _FusionCudaArray
from cupy.cuda cimport runtime


@util.memoize()
def _cuda_compile(preamble, name, cuda_params, cuda_body):
    code = string.Template(
        '${preamble}\n'
        'extern "C" __global__ void ${name}(${cuda_params}) {\n'
        '${cuda_body}\n'
        '}\n'
    ).substitute(
        preamble=preamble,
        name=name,
        cuda_params=cuda_params,
        cuda_body=cuda_body)
    module = compile_with_cache(code)

    # (For contributers) We can view the whole generated CUDA code
    # by uncommenting the following line.

    # print(code)

    return module.get_function(name)


cdef ndarray _reduce_dims_core(ndarray array):
    """Reduce the number of dimensions of the input array.
    """
    cdef vector.vector[Py_ssize_t] shape, strides, new_shape, new_strides
    cdef Py_ssize_t back
    shape = array._shape
    strides = array._strides
    new_shape.push_back(shape.back())
    shape.pop_back()
    new_strides.push_back(strides.back())
    strides.pop_back()
    while shape.size() > 0:
        if new_shape.back() * new_strides.back() == strides.back():
            new_shape[new_shape.size() - 1] *= shape.back()
        else:
            new_shape.push_back(shape.back())
            new_strides.push_back(strides.back())
        shape.pop_back()
        strides.pop_back()
    new_shape = new_shape[::-1]
    new_strides = new_strides[::-1]
    # TODO(asi1024): Use ``_ndarray_init`` after cupy/cupy#2701 is merged.
    return ndarray(new_shape, array.dtype, array.data, new_strides)


cdef class FusedKernel(object):
    cdef:
        readonly object shape_constraints

        readonly str _name
        readonly list _params
        readonly int _return_size
        readonly str _submodule_code
        readonly str _cuda_body
        readonly dict _cuda_params_memo
        readonly list _block_strides

        readonly list _reduction_in_array
        readonly list _reduction_out_array
        readonly vector.vector[bint] _is_base
        readonly list _dtypes
        readonly vector.vector[Py_ssize_t] _input_order
        readonly vector.vector[Py_ssize_t] _view_of
        readonly vector.vector[Py_ssize_t] _out_params

    def __init__(
            self, name, op_list, cuda_body, params, return_size,
            submodule_code, shape_constraints):
        self.shape_constraints = shape_constraints

        self._name = name
        self._params = sorted(params, key=lambda x: x.serial_number)
        self._submodule_code = submodule_code
        self._cuda_body = cuda_body
        self._cuda_params_memo = {}

        if return_size == 'none':
            self._return_size = -1
            self._out_params.resize(0)
        elif return_size == 'single':
            self._return_size = -2
            self._out_params.resize(1)
        else:
            self._return_size = return_size
            self._out_params.resize(return_size)

        for p in self._params:
            assert isinstance(p, _FusionCudaVarBase)

        array_dict = {}
        self._reduction_in_array = []
        self._reduction_out_array = []
        self._dtypes = []

        for i, p in enumerate(self._params):
            view_of = -1
            input_order = -1
            if p.input_order is not None:
                input_order = p.input_order
            if isinstance(p, _FusionCudaArray):
                if p._view_of is not None:
                    view_of = array_dict[p._view_of.key()]
                if p.is_output:
                    self._out_params[p.output_order] = i
            array_dict[p.key()] = i
            self._is_base.push_back(p.is_base)
            self._dtypes.append(_dtype.get_dtype(p.dtype))
            self._input_order.push_back(input_order)
            self._view_of.push_back(view_of)

        self._block_strides = []

        for op in op_list:
            if isinstance(op, _fusion_op._FusionReductionOp):
                self._reduction_in_array.append(
                    array_dict[op.in_params.item().key()])
                self._reduction_out_array.append(
                    array_dict[op.out_params.item().key()])
                self._block_strides.append(
                    'int {}'.format(op.block_stride_name))

    def get_shapes_of_kernel_params(self, tuple args):
        """Returns the shapes of paramters passed to kern.linear_launch.
        """
        cdef dict dim_map = {}
        cdef list kernel_param_shapes = []
        cdef int input_order
        cdef int axis
        cdef list shape

        for input_order in range(len(args)):
            arg = args[input_order]
            if isinstance(arg, ndarray):
                shape = arg._shape
                for axis in range(len(shape)):
                    dim_map[(input_order << 8) | axis] = shape[axis]

        for param in self._params:
            shape = []
            if isinstance(param, _FusionCudaArray):
                ashape = param.ashape
                for axis in range(len(ashape)):
                    dim = ashape[axis]
                    if not isinstance(dim, int):
                        dim = dim_map[dim._value]
                    shape.append(dim)
            kernel_param_shapes.append(shape)
        return kernel_param_shapes

    cdef list _get_ndarray_list(self, tuple args, list shapes):
        """Get the list of ndarray corresponding to ``self._params``.
        """
        cdef list ndarray_list = []
        cdef list params = self._params
        cdef int i
        for i in range(len(params)):
            param = params[i]
            shape = shapes[i]
            if self._input_order[i] >= 0:
                array = args[<Py_ssize_t>self._input_order[i]]
            elif isinstance(param, _FusionCudaScalar):
                array = None
            elif self._is_base[i]:
                # TODO(asi1024): Use ``_ndarray_init`` after cupy/cupy#2701
                # is merged.
                array = ndarray(shape, self._dtypes[i])
            else:
                view_of = ndarray_list[<Py_ssize_t>self._view_of[i]]
                if param.is_broadcast:
                    array = _manipulation.broadcast_to(view_of, shape)
                elif param.slice_key is not None:
                    array = view_of[param.slice_key]
                elif param.rotate_axis is not None:
                    axis_permutes = list(param.rotate_axis)
                    for i in range(param.ndim):
                        if i not in param.rotate_axis:
                            axis_permutes.append(i)
                    axis_permutes = tuple(axis_permutes)
                    array = _manipulation._transpose(view_of, axis_permutes)
                else:
                    assert False
            ndarray_list.append(array)
        return ndarray_list

    cdef object _get_return_value(self, list ndarray_list):
        """Get the return value of ``self.execute``.
        """
        cdef int i

        if self._return_size == -1:
            return None

        if self._return_size == -2:
            return ndarray_list[<Py_ssize_t>self._out_params[0]]

        return tuple([
            ndarray_list[<Py_ssize_t>self._out_params[i]]
            for i in range(self._return_size)
        ])

    cdef tuple _get_kernel_size(self, list ndarray_list):
        """Calculate the numnber of contiguous blocks in non-reduction axes
        of input arrays, and set them to ``self._contiguous_size``.
        """
        cdef ndarray in_array, out_array
        cdef Py_ssize_t block_size, block_stride, contiguous_size

        cdef list block_strides = []
        cdef Py_ssize_t kern_size = 1

        for i in range(len(self._params)):
            array = ndarray_list[i]
            if isinstance(array, ndarray):
                kern_size = max(kern_size, <Py_ssize_t>array.size)

        if len(self._reduction_in_array) == 0:
            return [], 128, 0, kern_size

        block_size = 256 if runtime._is_hip_environment else 512
        for i in range(len(self._reduction_in_array)):
            in_array = ndarray_list[self._reduction_in_array[i]]
            out_array = ndarray_list[self._reduction_out_array[i]]

            # TODO(asi1024): Fix block strides for performance.
            contiguous_size = 1
            itemsize = in_array.dtype.itemsize
            for i in range(out_array.ndim):
                if in_array.strides[-i-1] != contiguous_size * itemsize:
                    break
                contiguous_size *= in_array.shape[-i-1]
            contiguous_size = min(contiguous_size, 32)

            reduce_block_size = max(1, in_array.size // max(1, out_array.size))
            block_stride = max(contiguous_size, block_size // reduce_block_size)
            block_stride = internal.clp2(block_stride // 2 + 1)  # floor
            block_strides.append(block_stride)

        kern_size = (kern_size + block_size - 1) // block_size * block_size
        shared_mem = block_size * 32  # max bytesize of reduce_ctype.
        return block_strides, block_size, shared_mem, kern_size

    cdef tuple _reduce_dims(self, list ndarray_list):
        """Reduce number of dimensions of ndarrays and returns the cache key.
        """
        cdef list params = self._params
        cdef list ndims = []
        cdef int i

        for i in range(len(params)):
            param = params[i]
            array = ndarray_list[i]
            if param.ndim < 2:
                continue
            array = _reduce_dims_core(array)
            ndarray_list[i] = array
            ndims.append(array.ndim)

        return tuple(ndims)

    cdef list _get_inout_args(self, tuple args, list ndarray_list):
        """Get the arguments passed to ``kern.linear_launch``.
        """
        cdef list params = []
        cdef list indexers = []
        cdef list block_strides = []
        cdef int kern_size = 1

        for i in range(len(self._params)):
            array = ndarray_list[i]
            if isinstance(array, ndarray):
                params.append(array)
                indexers.append(Indexer(array.shape))
                kern_size = max(kern_size, array.size)
            elif self._input_order[i] >= 0:
                scalar = args[<Py_ssize_t>self._input_order[i]]
                params.append(_scalar.convert_scalar(scalar, False))

        return params + indexers

    cdef str _get_cuda_params(self, tuple key, list ndarray_list):
        """Get a string of parameters of CUDA main function code.
        """
        cdef int i

        if key in self._cuda_params_memo:
            return self._cuda_params_memo[key]

        cuda_params = []
        indexers = []

        for i in range(len(self._params)):
            a = self._params[i]
            if isinstance(a, _FusionCudaArray):
                ndim = ndarray_list[i].ndim
                cuda_params.append(a.format(
                    'CArray<${type}, ${ndim}> ${var}', ndim=ndim))
                indexers.append(
                    a.format('CIndexer<${ndim}> ${indexer}', ndim=ndim))
            elif isinstance(a, _FusionCudaScalar):
                if a.const_value is None:
                    cuda_params.append(a.format('${type} ${var}'))
            else:
                raise TypeError('Unknown type {}.'.format(type(a)))

        ret = cuda_params + indexers + self._block_strides
        ret = ', '.join(ret)
        self._cuda_params_memo[key] = ret
        return ret

    def execute(self, tuple args, list shapes):
        ndarray_list = self._get_ndarray_list(args, shapes)
        ret = self._get_return_value(ndarray_list)
        size_info = self._get_kernel_size(ndarray_list)
        block_strides, block_size, shared_mem, size = size_info
        reduce_key = self._reduce_dims(ndarray_list)
        inout_args = self._get_inout_args(args, ndarray_list)
        cuda_params = self._get_cuda_params(reduce_key, ndarray_list)
        # assert len(cuda_params) == len(inout_args) + len(block_strides)
        kern = _cuda_compile(
            self._submodule_code, self._name, cuda_params, self._cuda_body)
        kargs = inout_args + block_strides
        kern.linear_launch(size, kargs, shared_mem, block_size)
        return ret
