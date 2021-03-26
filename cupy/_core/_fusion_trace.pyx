import numpy

from cupy._core import _kernel
from cupy._core import _reduction
from cupy._core import core
from cupy._core._fusion_interface import _VariableProxy
from cupy._core._fusion_interface import _ArrayProxy
from cupy._core import _fusion_thread_local
from cupy._core import _fusion_variable
from cupy._core._fusion_variable import _AbstractDim
from cupy._core._fusion_variable import _TraceScalar
from cupy._core._fusion_variable import _TraceArray
from cupy._core._fusion_variable import _VariableSet
from cupy._core import _fusion_op
from cupy._core import _fusion_optimization
from cupy._core cimport internal


_thread_local = _fusion_thread_local.thread_local
_accepted_types = (int, float, bool, complex, numpy.generic)


cdef class _ShapeConstraints:
    """A data structure that manages the conditions between the shapes.
    """

    cdef:
        # A list of tuple of _AbstractDim and _AbstractDim which represents
        # the equality between dimensions.
        readonly list eq_constraints
        # A list of tuple of _AbstractDim and int which is an associative list
        readonly list const_constraints

    def __init__(self):
        self.eq_constraints = []
        self.const_constraints = []

    def add_eq_constraint(self, x, y):
        """Add a constraint: x == y.
        """
        _fusion_thread_local.check_not_runtime()
        assert isinstance(x, (_AbstractDim, int))
        assert isinstance(y, (_AbstractDim, int))
        x = self.evaluate(x)
        y = self.evaluate(y)
        if x == y:
            return
        if isinstance(x, _AbstractDim) and isinstance(y, _AbstractDim):
            self.eq_constraints.append((x, y))
        elif isinstance(x, _AbstractDim) and not isinstance(y, _AbstractDim):
            self.add_const_constraint(x, y)
        elif not isinstance(x, _AbstractDim) and isinstance(y, _AbstractDim):
            self.add_const_constraint(y, x)
        else:
            assert False

    def add_const_constraint(self, x, value):
        """Add a constraint: x == value.
        """
        _fusion_thread_local.check_not_runtime()
        assert isinstance(x, (_AbstractDim, int))
        assert isinstance(value, int)
        x = self.evaluate(x)
        if isinstance(x, _AbstractDim):
            self.const_constraints.append((x, value))
        else:
            assert x == value

    def evaluate(self, x):
        """Substitute repeatedly from the equalities.
        """
        _fusion_thread_local.check_not_runtime()
        assert isinstance(x, (_AbstractDim, int))
        for src, dest in self.eq_constraints + self.const_constraints:
            if isinstance(x, int):
                return x
            if x == src:
                x = dest
        return x

    # Used in runtime.
    def satisfy(self, dict dim_map):
        """Check if the given dicionary satisfies the constraints.

        Args:
            dim_map (dict):
                A dictionary with keys of _AbstractDim type and
                values of int type.
        """
        for a, b in self.eq_constraints:
            if dim_map[a] != dim_map[b]:
                return False
        for a, b in self.const_constraints:
            if dim_map[a] != b:
                return False
        return True


def _guess_routine(func, args, dtype):
    assert isinstance(func, (_kernel.ufunc, _reduction._SimpleReductionKernel))

    # Feeds dummy arguments with appropriate dtypes passed to `guess_routine`.
    dummy_args = []
    for x in args:
        if isinstance(x, _TraceScalar):
            obj = x.dtype.type(0)
        else:
            assert isinstance(x, _TraceArray)
            obj = core.ndarray((0,), x.dtype)
        dummy_args.append(obj)

    op = func._ops.guess_routine(
        func.name, func._routine_cache, dummy_args, dtype, None)
    return op.get_in_dtypes(), op.get_out_dtypes(), op.routine


def _base(array):
    """Returns the base array object of given array.
    """
    assert isinstance(array, core.ndarray)
    return array if array.base is None else array.base


class _VariableCoordinator:
    """Variable constuct manager.

    This class calls ``_TraceArray`` or ``_TraceScalar`` internally
    with unique serial numbers and returns the variable object. In
    ``TraceImpl`` class, a method of ``history.vc``, which is of
    ``_VariableConduductor`` class, should be called instead of
    ```_TraceArray.__init__`` or ``_TraceScalar.__init__``.
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
        memory = _fusion_variable._MemorySpace(
            self._memory_number, serial_number)
        self._serial_number += 1
        self._memory_number += 1

        ret = var_module(memory, serial_number, dtype, **kwargs)
        memory.is_input = ret.is_input
        return self._normalize_variable(ret)

    def generate_new_array(self, dtype, rshape, ashape, input_index=None):
        """Generate new _TraceArray object with a new memory space.
        """
        ret = self._generate_new_variable(
            _TraceArray,
            dtype, rshape=rshape, ashape=ashape, input_index=input_index)
        ret.memory.base_ashape = ret.ashape
        return ret

    def generate_new_scalar(self, dtype, **kwargs):
        """Generate new _TraceScalar object with a new memory space.
        """
        return self._generate_new_variable(_TraceScalar, dtype, **kwargs)

    def make_view(self, var, **kwargs):
        assert isinstance(var, _TraceArray)
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
        assert isinstance(var, _TraceArray)
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


class TraceImpl:
    """Emit a fused kernel from the given target function.
    """

    def __init__(self):
        self.vc = _VariableCoordinator()
        self.shape_constraints = _ShapeConstraints()
        self.op_list = []

    @staticmethod
    def _make_interface(x):
        """Returns an _array or a _scalar object which packs the given value.
        """
        if x is None:
            return None
        assert isinstance(x, _fusion_variable._TraceVariable)
        return x.as_interface()

    def _unwrap_interface(self, x, *, allow_none=False):
        """Returns ``_TraceVariable`` object from the input.
        """
        if allow_none and x is None:
            return None
        if isinstance(x, _VariableProxy):
            return x.content
        if isinstance(x, _accepted_types):
            dtype = numpy.dtype(type(x))
            return self.vc.generate_new_scalar(dtype, const_value=x)
        if isinstance(x, (numpy.ndarray, core.ndarray)):
            raise TypeError('Concrete ndarray is not supported in fusion.')
        raise TypeError('{} type is not supported'.format(type(x)))

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

        in_params = [self._unwrap_interface(x) for x in args[:nin]]
        out_params = [
            self._unwrap_interface(x, allow_none=True)
            for x in args[nin:] + (kwargs.pop('out', None),)
            if x is not None
        ]
        params = in_params + out_params

        if len(kwargs) > 0:
            raise TypeError('Wrong arguments {}'.format(kwargs))
        if len(in_params) != nin or len(out_params) > nout:
            raise ValueError('Invalid number of arguments')
        if not all([isinstance(v, _TraceArray) for v in out_params]):
            raise TypeError('Return arrays must be of ArrayType')

        # Check for inplace operation.
        for i, out_param1 in enumerate(out_params):
            for out_param2 in out_params[:i]:
                if out_param1.memory == out_param2.memory:
                    # NumPy does not raise this error.
                    raise ValueError('Outputs of ufunc must not share memory')

        # Copy the input array data before the operation when the input array
        # shares the same memory area with an output array.
        for i, in_param in enumerate(in_params):
            should_copy = any([
                in_param.memory == out_param.memory and in_param != out_param
                for out_param in out_params
            ])
            if should_copy:
                in_params[i] = self._unwrap_interface(
                    self.call_ufunc(
                        core.elementwise_copy,
                        self._make_interface(in_param)))

        # Broadcast shapes
        out_rshape = internal._broadcast_shapes([p.rshape for p in params])
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
            if isinstance(p, _TraceArray) and p.rshape != out_rshape:
                # Broadcst input if needed.
                in_params[i] = self.vc.broadcast_to(p, out_ashape, out_rshape)

        # Get operation code from dtypes.
        in_dtypes, out_dtypes, expr = _guess_routine(
            ufunc, in_params, dtype)

        # Make output arrays.
        ret = []
        for i in range(nout):
            if i >= len(out_params):
                # Omitted output.
                out_pvar = self.vc.generate_new_array(
                    out_dtypes[i], out_rshape, out_ashape)
                out_params.append(out_pvar)
                ret.append(out_pvar)
            elif isinstance(out_params, _TraceScalar):
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

        # Register Op.
        name = ufunc.name + '_' + str(len(self.op_list))
        ufunc_routine = _fusion_op._UfuncRoutine(
            name, ufunc, expr, in_params, out_params, in_dtypes + out_dtypes)
        op = _fusion_op._ElementwiseTraceOp(
            [ufunc_routine], in_params, out_params, out_ashape)
        self.op_list.append(op)

        # Returns.
        assert len(ret) > 0
        if len(ret) == 1:
            return self._make_interface(ret[0])
        else:
            return tuple([self._make_interface(x) for x in ret])

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
        in_param = self._unwrap_interface(a)

        if not isinstance(in_param, _TraceArray):
            raise NotImplementedError(
                'Reduction for scalar arguments is not supported.')

        axes = internal._normalize_axis_indices(axis, in_param.ndim)

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

        # Make an output array.
        if out is None:
            # Omitted output.
            out_param = self.vc.generate_new_array(
                out_dtype, out_rshape, out_ashape)
        else:
            out_param = self._unwrap_interface(out)
            if out_param.rshape != out_rshape:
                raise ValueError(
                    'Shape of specified output variable is not consistent '
                    'with reduced shape.')

        # Register Op.
        name = 'reduce{}'.format(len(self.op_list))
        op = _fusion_op._ReductionTraceOp(
            name, reduce_func, expr, in_param, out_param, axes)
        self.op_list.append(op)

        # Returns.
        return self._make_interface(out_param)

    def call_indexing(self, in_param, indices):
        """Call indexing routines.
        """
        in_param = self._unwrap_interface(in_param)

        if not isinstance(indices, tuple):
            indices = (indices,)

        for x in indices:
            if isinstance(indices, (list, _TraceArray)):
                # Advanced indexing
                raise NotImplementedError(
                    'Advanced indexing is not supported, currently.')

            if not (isinstance(x, (int, slice)) or x is None or x is Ellipsis):
                raise IndexError(
                    'Indices must be integers, slices, ellipsis, None or '
                    'integer or boolean arrays.')

        # Basic indexing
        out_param = self.vc.indexing(in_param, indices)
        return self._make_interface(out_param)

    def trace(self, func, args):
        """Call ``self.func`` with _TraceVariable arguments.

        Returns:
            out_params(list of _TraceVariable): The list of outputs.
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
                    assert isinstance(var, _TraceArray)
                elif base_id in memory_dict:
                    # The is an array which shares the same memory.
                    base = in_params[memory_dict[base_id]]
                    assert isinstance(base, _TraceArray)
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
        inputs = [self._make_interface(x) for x in in_params]
        output = func(*inputs)

        # Register output variables.
        if output is None:
            return_size = 'none'
            out_params = []
        elif isinstance(output, _ArrayProxy):
            return_size = 'single'
            out_params = [self._unwrap_interface(output, allow_none=True)]
        elif isinstance(output, tuple):
            if all(isinstance(x, _ArrayProxy) for x in output):
                return_size = len(output)
                out_params = [
                    self._unwrap_interface(x, allow_none=True) for x in output]
            else:
                raise ValueError(
                    'The all elements of return value of fused function '
                    'must be of _ArrayProxy type.'
                )
        else:
            raise ValueError(
                'The return value of fused functions must be `None`, '
                'ndarray or a tuple of ndarays.'
            )

        for output_index, out_param in enumerate(out_params):
            assert isinstance(out_param, _TraceArray)
            out_param.output_index = output_index
            out_param.memory.is_output = True

        return out_params, return_size


def _get_ancestors_of_trace_variable(var):
    if var is None:
        return _VariableSet()
    res = _VariableSet(var)
    if isinstance(var, _TraceArray):
        res += _get_ancestors_of_trace_variable(var._view_of)
    return res


class _TraceResult:

    def __init__(self, op_list, params, return_size, shape_constraints):
        self.op_list = op_list
        self.params = params
        self.return_size = return_size
        self.shape_constraints = shape_constraints


def trace(func, args):
    history = TraceImpl()

    try:
        _thread_local.history = history

        # Call `func(args)` and update `op_list`.
        out_params, return_size = history.trace(func, args)
    finally:
        _thread_local.history = None

    op_list = history.op_list
    shape_constraints = history.shape_constraints
    all_variables = history.vc.all_variables

    op_list = _fusion_optimization.optimize(
        op_list, all_variables, shape_constraints)

    # Make info passed to FusedKernel.
    kernel_params = _VariableSet()
    for p in out_params:
        kernel_params += _get_ancestors_of_trace_variable(p)
    for op in op_list:
        for p in op.in_params + op.out_params:
            kernel_params += _get_ancestors_of_trace_variable(p)
    kernel_params = list(kernel_params)

    # used in mock tests.
    history.kernel_params = kernel_params
    history.op_list = op_list

    return _TraceResult(op_list, kernel_params, return_size, shape_constraints)
