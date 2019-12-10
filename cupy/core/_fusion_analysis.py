import six

import numpy

from cupy.core import _errors
from cupy.core import _kernel
from cupy.core import core
from cupy.core import _fusion_interface
from cupy.core import _fusion_variable
from cupy.core._fusion_variable import _FusionCudaScalar
from cupy.core._fusion_variable import _FusionCudaArray
from cupy.core._fusion_variable import _FusionVariableSet
from cupy.core import _fusion_shape
from cupy.core import _fusion_device_func
from cupy.core import _fusion_op
from cupy.core._fusion_emit_code import _CodeBlock
from cupy.core import _fusion_runtime
from cupy.core import _fusion_optimization


_accepted_types = six.integer_types + (float, bool, complex, numpy.generic)


def _broadcast_shapes(shapes):
    """Returns the braodcasted shapes.
    """
    out_ndim = max([len(shape) for shape in shapes])
    shapes = [(1,) * (out_ndim - len(shape)) + shape for shape in shapes]

    result_shape = []
    for dims in zip(*shapes):
        dims = [dim for dim in dims if dim != 1]
        out_dim = 1 if len(dims) == 0 else dims[0]
        if any([dim != out_dim for dim in dims]):
            raise ValueError(
                'Operands could not be broadcast together with shapes' +
                ' '.join([str(shape) for shape in shapes]))
        result_shape.append(out_dim)

    return tuple(result_shape)


def _normalize_axis(axes, ndim):
    """Normalize axes. Returns a tuple of ints in [0, ndim) range.
    """
    if axes is None:
        axes = tuple(range(ndim))
    elif not isinstance(axes, tuple):
        axes = axes,

    res = []
    for axis in axes:
        if not isinstance(axis, int):
            raise TypeError(
                '{} cannot be interpreted as an integer.'.format(type(axis)))
        if not (-ndim <= axis < ndim):
            raise _errors._AxisError(
                'axis {} is out of bounds for array of dimension {}.'.format(
                    axis, ndim))
        axis %= ndim
        if axis in res:
            raise ValueError('Duplicate value in \'axis\'')
        res.append(axis % ndim)

    return tuple(sorted(res))


_kind_score = {
    'b': 0,
    'u': 1,
    'i': 1,
    'f': 2,
    'c': 2,
}


def _base(array):
    assert isinstance(array, core.ndarray)
    return array if array.base is None else array.base


def _get_ufunc_dtypes_op(ufunc, in_params, dtype=None):
    """Get the dtypes of parameters and operation code to be emitted.

    Args:
        ufunc(ufunc): The ufunc.
        in_params(list of {_FusionCudaScalar or _FusionCudaArray}): The inputs.
        dtype(dtype): The dtype specified by users.

    Returns: A tuple of size 3. The first element indicates the dtypes of the
        inputs, the second element indicates the dtypes of the outputs, and the
        third element is the CUDA device function body of string type.
    """
    if dtype is not None:
        raise NotImplementedError(
            'dtype keyword argument is not supported, currently.')

    # Corresponds to _checkshould_use_min_scalar in _kernel.pyx.
    # This function decides which typecast rule to use.
    max_array_kind = -2
    max_scalar_kind = -1
    for p in in_params:
        kind = _kind_score[p.dtype.kind]
        if isinstance(p, _FusionCudaArray):
            max_array_kind = max(max_array_kind, kind)
        elif isinstance(p, _FusionCudaScalar):
            max_scalar_kind = max(max_scalar_kind, kind)
        else:
            assert False
    _should_use_min_scalar = (
        max_scalar_kind != -1 and max_array_kind >= max_scalar_kind)

    for in_dtypes, out_dtypes, op in ufunc._ops:
        in_dtypes = [numpy.dtype(t) for t in in_dtypes]
        out_dtypes = [numpy.dtype(t) for t in out_dtypes]
        can_cast = True
        for param, dtype in zip(in_params, in_dtypes):
            assert isinstance(param, (_FusionCudaScalar, _FusionCudaArray))
            if _should_use_min_scalar and isinstance(param, _FusionCudaScalar):
                if param.const_value is None:
                    # This typecast is not safe.
                    # The result of a typecast of an element-wise operation
                    # between a numpy ndarray and a numpy scalar is not
                    # decidable statically, because it depends on the value
                    # of the scalar variable.
                    cast_from = param.dtype.type(0)
                else:
                    cast_from = param.const_value
            else:
                cast_from = param.dtype
            if not numpy.can_cast(cast_from, dtype):
                can_cast = False
                break
        if can_cast:
            return in_dtypes, out_dtypes, op

    raise TypeError('Invalid type cast in \'{}\' dtypes={}'.format(
        ufunc.name, [param.dtype for param in in_params]))


def _get_reduction_dtypes_op(reduction, in_param, dtype=None):
    """Get the dtypes of parameters and operation code to be emitted.

    Args:
        reduction(_SimpleReductionKernel): The reduction object.
        in_param(_FusionCudaScalar or _FusionCudaArray): The input.
        dtype(dtype): The dtype specified by users.

    Returns: Similar to ``_get_ufunc_dtypes_op``.
    """
    assert isinstance(in_param, _FusionCudaArray)

    for in_dtypes, out_dtypes, op in reduction._ops:
        in_dtype, = [numpy.dtype(t) for t in in_dtypes]
        out_dtype, = [numpy.dtype(t) for t in out_dtypes]

        if dtype is None:
            if numpy.can_cast(in_param.dtype, in_dtype):
                return in_dtypes, out_dtypes, out_dtype, op
        else:
            if numpy.can_cast(dtype, in_dtype):
                return in_dtypes, out_dtypes, out_dtype, op

    raise TypeError('Invalid type cast in \'{}\' dtype={}'.format(
        reduction.name, in_param.dtype))


class _VariableConductor(object):
    """Variable constuct manager.

    This class calls ``_FusionCudaArray`` or ``_FusionCudaScalar`` internally
    with unique serial numbers and returns the variable object. In
    ``_FusionHistory`` class, a method of ``history.vc``, which is of
    ``_VariableConduductor`` class, should be called instead of
    ```_FusionCudaArray.__init__`` or ``_FusionCudaScalar.__init__``.
    """

    def __init__(self):
        self._memory_number = 0
        self._serial_number = 0
        self._variables_dict = {}

    def _normalize_variable(self, var):
        """If the input variable is already generated previously, returns it.
        """
        key = var.key()
        if key not in self._variables_dict:
            self._variables_dict[key] = var
        return self._variables_dict[key]

    def _generate_new_variable(self, var_module, dtype, **kwargs):
        serial_number = self._serial_number
        memory = _fusion_variable._FusionMemorySpace(
            self._memory_number, serial_number)
        self._serial_number += 1
        self._memory_number += 1

        ret = var_module(memory, serial_number, dtype, **kwargs)
        memory.is_input = ret.is_input
        return self._normalize_variable(ret)

    def generate_new_array(self, dtype, rshape, ashape, input_order=None):
        """Generate new _FusionCudaArray object with a new memory space.
        """
        ret = self._generate_new_variable(
            _FusionCudaArray,
            dtype, rshape=rshape, ashape=ashape, input_order=input_order)
        ret.memory.base_ashape = ret.ashape
        return ret

    def generate_new_scalar(self, dtype, **kwargs):
        """Generate new _FusionCudaArray object with a new memory space.
        """
        return self._generate_new_variable(_FusionCudaScalar, dtype, **kwargs)

    def make_view(self, var, **kwargs):
        assert isinstance(var, _FusionCudaArray)
        serial_number = self._serial_number
        self._serial_number += 1
        ret = var.make_view(serial_number, **kwargs)
        return self._normalize_variable(ret)

    def broadcast_to(self, var, ashape, rshape):
        """Make a view of the input array with the given shape.
        """
        return self.make_view(
            var, ashape=ashape, rshape=rshape, broadcasted_from=var)

    def rotate_with_axis(self, var, axis):
        """Make a view of an array by rotating ``var`` with given axis.
        """
        assert isinstance(var, _FusionCudaArray)
        return self.make_view(var, rotated_from=var, axis=axis)

    def index_with_int(self, var, key):
        """Make a view of an array. by indexing ``var`` with given integer.
        """
        return self.index_with_tuple(var, (key,))

    def index_with_tuple(self, var, key):
        """Make a view of an array. by indexing ``var`` with given tuple.
        """
        for k in key:
            if not isinstance(k, int):
                raise IndexError(
                    'Cannot subscript by type {}.'.format(type(k)))
        if len(key) > var.ndim:
            raise IndexError('too many indices for array.')

        ashape = var.ashape[len(key):]
        rshape = var.rshape[len(key):]

        return self.make_view(
            var, indexed_from=var, index_key=key,
            ashape=ashape, rshape=rshape)

    @property
    def all_variables(self):
        """Returns the list of all variables this class emitted.
        """
        return list(self._variables_dict.values())


class _FusionHistory(object):
    """History of operation exectuted in the target function of fusion.
    """

    def __init__(self, name):
        self.name = name
        self.vc = _VariableConductor()
        self.shape_constraints = _fusion_shape._ShapeConstraints()
        self.op_list = []
        self.submodules = {}

    @staticmethod
    def _to_interface(x):
        """Returns an _array or a _scalar object which packs the given value.
        """
        if x is None:
            return None
        assert isinstance(x, _fusion_variable._FusionCudaVarBase)
        return x.as_interface()

    def _from_arraylike_interface(self, x):
        """Returns ``_FusionCuda{Array/Scalar}`` object from the input.
        """
        if isinstance(x, _fusion_interface._FusionVariableInterfaceBase):
            return x.content
        if isinstance(x, _accepted_types):
            dtype = numpy.dtype(type(x))
            return self.vc.generate_new_scalar(dtype, const_value=x)
        if isinstance(x, (numpy.ndarray, core.ndarray)):
            raise TypeError('Concrete ndarray is not supported in fusion.')
        raise TypeError('{} type is not supported'.format(type(x)))

    def _from_interface(self, x):
        """Returns ``_FusionCuda{Array/Scalar}`` object or ``None``.
        """
        if x is None:
            return None
        return self._from_arraylike_interface(x)

    def call_ufunc(self, ufunc, *args, **kwargs):
        """Register an elementwise operation with the given parameters.

        Args:
            ufunc(_kernel.ufunc): The ufunc to operate.
            args(tuple): The arguments.
            kwargs(dict): The keyword arguments.
        """

        assert isinstance(ufunc, _kernel.ufunc)

        # Parse Inputs.
        nin = ufunc.nin
        nout = ufunc.nout

        if 'out' in kwargs and len(args) > nin:
            raise ValueError(
                'cannot specify \'out\' as both a positional and '
                'keyword argument')

        in_params = [self._from_arraylike_interface(x) for x in args[:nin]]
        out_params = [
            self._from_interface(x)
            for x in args[nin:] + (kwargs.pop('out', None),)
            if x is not None
        ]
        params = in_params + out_params

        if len(kwargs) > 0:
            raise TypeError('Wrong arguments {}'.format(kwargs))
        if len(in_params) != nin or len(out_params) > nout:
            raise ValueError('Invalid number of arguments')
        if not all([isinstance(v, _FusionCudaArray) for v in out_params]):
            raise TypeError('Return arrays must be of ArrayType')

        # Check for inplace operation.
        for i, out_param1 in enumerate(out_params):
            for out_param2 in out_params[:i]:
                if out_param1.memory == out_param2.memory:
                    # NumPy does not raise this error.
                    raise ValueError('Outputs of ufunc must not share memory')

        for i, in_param in enumerate(in_params):
            should_copy = any([
                in_param.memory == out_param.memory and in_param != out_param
                for out_param in out_params
            ])
            if should_copy:
                in_params[i] = self._from_arraylike_interface(
                    self.call_ufunc(
                        core.elementwise_copy,
                        self._to_interface(in_param)))

        # Broadcast shapes
        out_rshape = _broadcast_shapes([p.rshape for p in params])
        out_ashape = [None for _ in range(len(out_rshape))]

        for p in params:
            for axis in range(-p.ndim, 0):
                if p.rshape[axis] == out_rshape[axis]:
                    out_ashape[axis] = p.ashape[axis]

        assert all([dim is not None for dim in out_ashape])
        out_ashape = tuple(out_ashape)

        # Broadcast input params and make their views.
        for i, p in enumerate(in_params):
            for axis in range(-p.ndim, 0):
                if p.rshape[axis] == out_rshape[axis]:
                    self.shape_constraints.add_eq_constraint(
                        p.ashape[axis], out_ashape[axis])
                elif p.rshape[axis] == 1:
                    self.shape_constraints.add_const_constraint(
                        p.ashape[axis], 1)
                else:
                    assert False
            if isinstance(p, _FusionCudaArray) and p.rshape != out_rshape:
                # Broadcst input if needed.
                in_params[i] = self.vc.broadcast_to(p, out_ashape, out_rshape)

        # Get operation code from dtypes.
        in_dtypes, out_dtypes, op = _get_ufunc_dtypes_op(ufunc, in_params)
        ret = []
        for i in range(nout):
            if i >= len(out_params):
                out_pvar = self.vc.generate_new_array(
                    out_dtypes[i], out_rshape, out_ashape)
                out_params.append(out_pvar)
                ret.append(out_pvar)
            elif isinstance(out_params, _FusionCudaScalar):
                raise TypeError('return arrays must be of ArrayType')
            elif out_params[i].rshape != out_rshape:
                raise ValueError(
                    'non-broadcastable output operand with shape {} '
                    'doesn\'t match the broadcast shape {}'.format(
                        out_params[i].rshape, out_rshape))
            elif numpy.can_cast(
                    out_dtypes[i], out_params[i].dtype, 'same_kind'):
                out_pvar = out_params[i]
                ret.append(out_pvar)
            else:
                raise TypeError(
                    'output (typecode \'{}\') could not be coerced '
                    'to provided output parameter (typecode \'{}\') '
                    'according to the casting rule '
                    '"same_kind"'.format(
                        out_dtypes[i].char, out_params[i].dtype.char))

        # Make submodule.
        name = ufunc.name + '_' + str(len(self.op_list))
        dtypes = in_dtypes + out_dtypes
        subm = _fusion_device_func._SubmoduleUfunc(
            name, ufunc, (op, dtypes), in_params, out_params)
        self.submodules[subm.name] = subm

        # Register Op.
        op = _fusion_op._FusionElementwiseOp(
            subm, in_params, out_params, out_ashape)
        self.op_list.append(op)

        # Returns.
        assert len(ret) > 0
        if len(ret) == 1:
            return self._to_interface(ret[0])
        else:
            return tuple([self._to_interface(x) for x in ret])

    def call_reduction(
            self, reduce_func, a, axis=None, dtype=None, out=None,
            keepdims=False):
        """Register a reduction operation with the given parameters.

        Args:
            reduce_func(_kernel._SimpleReductionKernel):
                The reduction function to operate.
            a(array_like): The input array.
            axis(int, tuple of int or None): The axis.
            dtype(numpy.dtype or None): The dtype
            out(_array or None): The output array.
        """

        assert isinstance(reduce_func, _kernel._SimpleReductionKernel)

        # Parse inputs.
        in_param = self._from_arraylike_interface(a)

        if not isinstance(in_param, _FusionCudaArray):
            raise NotImplementedError(
                'Reduction for scalar arguments is not supported.')

        axes = _normalize_axis(axis, in_param.ndim)

        if dtype is not None:
            dtype = numpy.dtype(dtype)

        if keepdims:
            raise NotImplementedError('keepdims is not supported.')

        # Determine the shape of out_param.
        out_ashape = tuple([
            d for axis, d in enumerate(in_param.ashape) if axis not in axes])
        out_rshape = tuple([
            d for axis, d in enumerate(in_param.rshape) if axis not in axes])

        # Rotate axes.
        # This condition is only for performance improvement,
        if not all([i == axis for i, axis in enumerate(axes)]):
            in_param = self.vc.rotate_with_axis(in_param, axes)

        # Get operation code from dtypes.
        in_dtypes, out_dtypes, op_dtype, reduce_op = _get_reduction_dtypes_op(
            reduce_func, in_param, dtype)
        if out is None:
            out_param = self.vc.generate_new_array(
                op_dtype, out_rshape, out_ashape)
        else:
            out_param = self._from_arraylike_interface(out)
            if out_param.rshape != out_rshape:
                raise ValueError(
                    'Shape of specified output variable is not consistent '
                    'with reduced shape.')

        # Make submodule.
        name = 'reduce{}'.format(len(self.op_list))
        subm = _fusion_device_func._SubmoduleReduction(
            name, reduce_func, reduce_op, [in_param], [out_param])
        self.submodules[subm.name] = subm

        # Register Op.
        op = _fusion_op._FusionReductionOp(subm, in_param, out_param, axes)
        self.op_list.append(op)

        # Returns.
        return self._to_interface(out_param)

    def call_indexing(self, in_param, index):
        """Call indexing routines.
        """
        in_param = self._from_arraylike_interface(in_param)
        assert isinstance(in_param, _FusionCudaArray)
        if isinstance(index, int):
            out_param = self.vc.index_with_int(in_param, index)
        elif isinstance(index, tuple):
            out_param = self.vc.index_with_tuple(in_param, index)
        else:
            raise NotImplementedError

        return self._to_interface(out_param)

    def _trace_target_function(self, func, args):
        """Call ``self.func`` with _FusionVariable arguments.

        Returns:
            out_params(list of _FusionVariable): The list of outputs.
            return_size(int or str): If ``return_size`` is of int type,
                it indicates the size of tuple of outputs.
                If `none`, the output is ``None`` and ``out_params`` is empty.
                If `single`, the output is single array and ``out_params``
                is a singleton list.

        During the function call, ``call_ufunc``, ``call_reduction`` and
        ``call_indexing`` are called internally.
        """

        # Register input variables.
        in_params = []
        array_dict = {}
        memory_dict = {}
        for input_order, arg in enumerate(args):
            if arg is None:
                var = None
            elif isinstance(arg, core.ndarray):
                arg_id = id(arg)
                base_id = id(_base(arg))
                if arg_id in array_dict:
                    # The array is already given as an input.
                    var = in_params[array_dict[arg_id]]
                    assert isinstance(var, _FusionCudaArray)
                elif base_id in memory_dict:
                    # The is an array which shares the same memory.
                    base = in_params[memory_dict[base_id]]
                    assert isinstance(base, _FusionCudaArray)
                    var = self.vc.make_view(base, input_order=input_order)
                else:
                    # Otherwise.
                    var = self.vc.generate_new_array(
                        arg.dtype, arg.shape, None, input_order=input_order)
                array_dict[arg_id] = input_order
                memory_dict[base_id] = input_order
            else:
                # Scalar input.
                dtype = numpy.dtype(type(arg))
                var = self.vc.generate_new_scalar(
                    dtype, input_order=input_order)
            in_params.append(var)

        # Call the target function.
        out_params = func(*[self._to_interface(x) for x in in_params])

        # Register output variables.
        if out_params is None:
            return_size = 'none'
            out_params = []
        elif isinstance(out_params, tuple):
            return_size = len(out_params)
            out_params = [self._from_interface(x) for x in out_params]
        else:
            return_size = 'single'
            out_params = [self._from_interface(out_params)]

        for output_order, out_param in enumerate(out_params):
            out_param.output_order = output_order
            out_param.memory.is_output = True

        return out_params, return_size

    def _get_ancestors(self, var):
        if var is None:
            return _FusionVariableSet()
        res = _FusionVariableSet(var)
        if isinstance(var, _FusionCudaArray):
            res += self._get_ancestors(var._view_of)
        return res

    def emit_kernel(self, func, args):
        # Call `func(args)` and update `op_list`.
        out_params, return_size = self._trace_target_function(func, args)
        self.op_list = _fusion_optimization.optimize(
            self.op_list, self.vc.all_variables, self.shape_constraints)

        # Make info passed to FusedKernel.
        kernel_params = _FusionVariableSet()
        for p in out_params:
            kernel_params += self._get_ancestors(p)
        for op in self.op_list:
            for p in op.in_params + op.out_params:
                kernel_params += self._get_ancestors(p)
        kernel_params = list(kernel_params)

        # Emit __device__ functions.
        preambles = list(set([s.preamble for s in self.submodules.values()]))
        submodules = [str(sub.emit_code()) for sub in self.submodules.values()]
        submodule_code = '\n'.join([
            s for s in preambles + submodules if s != ''])

        # Emit the function body of a __global__ function.
        codes = []
        for op in self.op_list:
            codes += op.emit_code().codes
            codes.append('__syncthreads();')

        cuda_body = str(_CodeBlock(codes))

        # This attribute is referred in mock tests.
        self.kernel_params = kernel_params

        return _fusion_runtime.FusedKernel(
            self.name, self.op_list, cuda_body, kernel_params, return_size,
            submodule_code, self.shape_constraints)
