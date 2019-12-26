import six

import numpy

from cupy.core import _errors
from cupy.core import _kernel
from cupy.core import _reduction
from cupy.core import core
from cupy.core import _fusion_interface
from cupy.core import _fusion_variable
from cupy.core._fusion_variable import _FusionCudaScalar
from cupy.core._fusion_variable import _FusionCudaArray
from cupy.core._fusion_variable import _FusionVariableSet
from cupy.core import _fusion_shape
from cupy.core import _fusion_device_func
from cupy.core import _fusion_op
from cupy.core import _fusion_emit_code
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


def _guess_routine(func, args, dtype):
    new_args = []
    for x in args:
        if isinstance(x, _FusionCudaScalar):
            new_args.append(x.dtype.type(0))
        else:
            new_args.append(core.ndarray((0,), x.dtype))
    op = func._ops.guess_routine(
        func.name, func._routine_cache, new_args, dtype, None)
    in_dtype = tuple([numpy.dtype(t) for t in op.in_types])
    out_dtype = tuple([numpy.dtype(t) for t in op.out_types])
    return in_dtype, out_dtype, op.routine


def _base(array):
    assert isinstance(array, core.ndarray)
    return array if array.base is None else array.base


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

    def generate_new_array(self, dtype, rshape, ashape, input_index=None):
        """Generate new _FusionCudaArray object with a new memory space.
        """
        ret = self._generate_new_variable(
            _FusionCudaArray,
            dtype, rshape=rshape, ashape=ashape, input_index=input_index)
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

    def indexing(self, var, indices):
        """Make a view of an array. by indexing ``var`` with given tuple.
        """
        skip = var.ndim - sum([isinstance(x, (int, slice)) for x in indices])
        it = 0
        ashape = []
        rshape = []

        if skip < 0:
            raise IndexError('Too many indices for array.')

        for index in indices:
            if isinstance(index, int):
                it += 1
            elif isinstance(index, slice):
                if not (index.start is None
                        and index.stop is None
                        and index.step in (1, -1, None)):
                    raise NotImplementedError(
                        'Only full range ``x[::]`` or reverse ``x[::-1]`` is '
                        'supported for basic slicing in CuPy fusion.')
                ashape.append(var.ashape[it])
                rshape.append(var.rshape[it])
                it += 1
            elif index is None:
                ashape.append(1)
                rshape.append(1)
            elif index is Ellipsis:
                ashape.extend(var.ashape[it:it + skip])
                rshape.extend(var.rshape[it:it + skip])
                it += skip

        ashape.extend(var.ashape[it:var.ndim])
        rshape.extend(var.rshape[it:var.ndim])

        return self.make_view(
            var, indexed_from=var, index_key=indices,
            ashape=tuple(ashape), rshape=tuple(rshape))

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
        dtype = kwargs.pop('dtype', None)

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
        in_dtypes, out_dtypes, expr = _guess_routine(ufunc, in_params, dtype)

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
            name, ufunc, (expr, dtypes), in_params, out_params)
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
            reduce_func(_reduction._SimpleReductionKernel):
                The reduction function to operate.
            a(array_like): The input array.
            axis(int, tuple of int or None): The axis.
            dtype(numpy.dtype or None): The dtype
            out(_array or None): The output array.
        """

        assert isinstance(reduce_func, _reduction._SimpleReductionKernel)

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
        _, (out_dtype,), expr = _guess_routine(reduce_func, [in_param], dtype)

        if out is None:
            out_param = self.vc.generate_new_array(
                out_dtype, out_rshape, out_ashape)
        else:
            out_param = self._from_arraylike_interface(out)
            if out_param.rshape != out_rshape:
                raise ValueError(
                    'Shape of specified output variable is not consistent '
                    'with reduced shape.')

        # Make submodule.
        name = 'reduce{}'.format(len(self.op_list))
        subm = _fusion_device_func._SubmoduleReduction(
            name, reduce_func, expr, [in_param], [out_param])
        self.submodules[subm.name] = subm

        # Register Op.
        op = _fusion_op._FusionReductionOp(subm, in_param, out_param, axes)
        self.op_list.append(op)

        # Returns.
        return self._to_interface(out_param)

    def call_indexing(self, in_param, indices):
        """Call indexing routines.
        """
        in_param = self._from_arraylike_interface(in_param)

        if not isinstance(indices, tuple):
            indices = (indices,)

        for x in indices:
            if isinstance(indices, (list, _FusionCudaArray)):
                # Advanced indexing
                raise NotImplementedError(
                    'Advanced indexing is not supported, currently.')

            if not (isinstance(x, (int, slice)) or x is None or x is Ellipsis):
                raise IndexError(
                    'Indices must be integers, slices, ellipsis, None or '
                    'integer or boolean arrays.')

        # Basic indexing
        out_param = self.vc.indexing(in_param, indices)
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
        for input_index, arg in enumerate(args):
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
                    var = self.vc.make_view(base, input_index=input_index)
                else:
                    # Otherwise.
                    var = self.vc.generate_new_array(
                        arg.dtype, arg.shape, None, input_index=input_index)
                array_dict[arg_id] = input_index
                memory_dict[base_id] = input_index
            else:
                # Scalar input.
                dtype = numpy.dtype(type(arg))
                var = self.vc.generate_new_scalar(
                    dtype, input_index=input_index)
            in_params.append(var)

        # Call the target function.
        inputs = [self._to_interface(x) for x in in_params]
        output = func(*inputs)

        # Register output variables.
        if output is None:
            return_size = 'none'
            out_params = []
        elif isinstance(output, _fusion_interface._ndarray):
            return_size = 'single'
            out_params = [self._from_interface(output)]
        elif isinstance(output, tuple):
            if all(isinstance(x, _fusion_interface._ndarray) for x in output):
                return_size = len(output)
                out_params = [self._from_interface(x) for x in output]
            else:
                raise ValueError(
                    'The all elements of return value of fused function '
                    'must be of _ndarray type.'
                )
        else:
            raise ValueError(
                'The return value of fused functions must be `None`, '
                'ndarray or a tuple of ndarays.'
            )

        for output_index, out_param in enumerate(out_params):
            assert isinstance(out_param, _FusionCudaArray)
            out_param.output_index = output_index
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
        submodule_code = '\n\n'.join([
            s for s in preambles + submodules if s != ''])

        # Emit the function body of a __global__ function.
        codes = []
        for op in self.op_list:
            codes.append(op.emit_code())
            codes.append('__syncthreads();')

        cuda_body = str(_fusion_emit_code._CodeBlock('', codes))

        # This attribute is referred in mock tests.
        self.kernel_params = kernel_params

        # print(cuda_body)

        return _fusion_runtime.FusedKernel(
            self.name, self.op_list, cuda_body, kernel_params, return_size,
            submodule_code, self.shape_constraints)
