import re

import numpy

import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal


# Signature parsing code and dimension accessing has been borrowed
# from dask
# https://github.com/dask/dask/blob/61b578f5a3ad88cbc6a8b9a73ce08c551bd969fa/dask/array/gufunc.py#L12-L55
_DIMENSION_NAME = r'\w+\?*'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*,?)?'.format(_DIMENSION_NAME)
_ARGUMENT = r'\({}\)'.format(_CORE_DIMENSION_LIST)
_INPUT_ARGUMENTS = '(?:{0:}(?:,{0:})*,?)?'.format(_ARGUMENT)
_OUTPUT_ARGUMENTS = '{0:}(?:,{0:})*'.format(
    _ARGUMENT
)  # Use `'{0:}(?:,{0:})*,?'` if gufunc-
# signature should be allowed for length 1 tuple returns
_SIGNATURE = '^{0:}->{1:}$'.format(_INPUT_ARGUMENTS, _OUTPUT_ARGUMENTS)


def _parse_gufunc_signature(signature):
    # The code has been modifyed from dask to support optional dimensions
    if not isinstance(signature, str):
        raise TypeError('Signature is not a string')

    if signature == '' or signature is None:
        raise ValueError('Signature cannot be empty')

    signature = signature.replace(' ', '')
    if not re.match(_SIGNATURE, signature):
        raise ValueError('Not a valid gufunc signature: {}'.format(signature))
    in_txt, out_txt = signature.split('->')
    ins = [tuple(x.split(',')) if x != '' else ()
           for x in in_txt[1:-1].split('),(')]
    outs = [tuple(y.split(',')) if y != '' else ()
            for y in out_txt[1:-1].split('),(')]
    # TODO(ecastill) multiple output support
    if len(outs) > 1:
        raise ValueError('Currently more than 1 output is not supported')
    return ins, outs


def _validate_normalize_axes(
    axes, axis, keepdims, input_coredimss, output_coredimss
):
    # This code credit goes to Dask
    # https://github.com/dask/dask/blob/61b578f5a3ad88cbc6a8b9a73ce08c551bd969fa/dask/array/gufunc.py#L58-L172
    nin = len(input_coredimss)
    nout = (
        1 if not isinstance(output_coredimss, list) else len(output_coredimss)
    )

    if axes is not None and axis is not None:
        raise ValueError(
            'Only one of `axis` or `axes` keyword arguments should be given')
    if axes and not isinstance(axes, list):
        raise ValueError('`axes` has to be of type list')

    # output_coredimss = output_coredimss if nout > 1 else [output_coredimss]
    filtered_core_dims = list(filter(len, input_coredimss))
    nr_outputs_with_coredims = len(
        [True for x in output_coredimss if len(x) > 0])

    if keepdims:
        if nr_outputs_with_coredims > 0:
            raise ValueError('`keepdims` can only be used for scalar outputs')
        output_coredimss = len(output_coredimss) * [filtered_core_dims[0]]

    core_dims = input_coredimss + output_coredimss
    if axis is not None:
        if not isinstance(axis, int):
            raise ValueError('`axis` argument has to be an integer value')
        if filtered_core_dims:
            cd0 = filtered_core_dims[0]
            if len(cd0) != 1:
                raise ValueError(
                    '`axis` can be used only, if one core dimension is present'
                )
            for cd in filtered_core_dims:
                if cd0 != cd:
                    raise ValueError(
                        'To use `axis`, all core dimensions have to be equal'
                    )

    # Expand defaults or axis
    if axes is None:
        if axis is not None:
            axes = [(axis,) if cd else tuple() for cd in core_dims]
        else:
            axes = [tuple(range(-len(icd), 0)) for icd in core_dims]

    axes = [(a,) if isinstance(a, int) else a for a in axes]

    if (
        (nr_outputs_with_coredims == 0)
        and (nin != len(axes))
        and (nin + nout != len(axes))
    ) or ((nr_outputs_with_coredims > 0) and (nin + nout != len(axes))):
        raise ValueError(
            'The number of `axes` entries is not equal the number'
            ' of input and output arguments')

    # Treat outputs
    output_axes = axes[nin:]
    output_axes = (
        output_axes
        if output_axes
        else [tuple(range(-len(ocd), 0)) for ocd in output_coredimss]
    )
    input_axes = axes[:nin]

    # Assert we have as many axes as output core dimensions
    for idx, (iax, icd) in enumerate(zip(input_axes, input_coredimss)):
        if len(iax) != len(icd):
            raise ValueError(
                f'The number of `axes` entries for argument #{idx}'
                ' is not equal the number of respective input core'
                ' dimensions in signature')
    if not keepdims:
        for idx, (oax, ocd) in enumerate(zip(output_axes, output_coredimss)):
            if len(oax) != len(ocd):
                raise ValueError(
                    f'The number of `axes` entries for argument #{idx}'
                    ' is not equal the number of respective output core'
                    ' dimensions in signature')
    else:
        if input_coredimss:
            icd0 = input_coredimss[0]
            for icd in input_coredimss:
                if icd0 != icd:
                    raise ValueError(
                        'To use `keepdims`, all core dimensions'
                        ' have to be equal')
            iax0 = input_axes[0]
            output_axes = [iax0 for _ in output_coredimss]

    return input_axes, output_axes


class _OpsRegister:
    '''
    Holds the ops for each dtypes signature like ('ff->f', func1)
    and allows to do look ups for these
    '''
    class _Op:
        def __init__(self, in_types, out_types, func):
            self.func = func
            self.in_types = tuple(numpy.dtype(i) for i in in_types)
            self.out_types = tuple(numpy.dtype(o) for o in out_types)
            self.sig_str = (''.join(
                in_t.char for in_t in self.in_types) + '->' + ''.join(
                    out_t.char for out_t in self.out_types))

    def __init__(self, signatures, default_func, nin, nout, name):
        self._default_func = default_func
        self._nin = nin
        self._nout = nout
        self._ops = self._process_signatures(signatures)
        self._name = name

    def _sig_str_to_tuple(self, sig):
        sig = sig.replace(' ', '')
        toks = sig.split('->')
        if len(toks) != 2:
            raise ValueError(f'signature {sig} for dtypes is invalid')
        else:
            ins, outs = toks
        return ins, outs

    def _process_signatures(self, signatures):
        ops = []
        for sig in signatures:
            if isinstance(sig, tuple):
                sig, op = sig
            else:
                op = self._default_func
            ins, outs = self._sig_str_to_tuple(sig)
            # Check the number of inputs and outputs matches the gufunc sig
            if len(ins) != self._nin:
                raise ValueError(
                    f'signature {sig} for dtypes is invalid number of inputs '
                    'is not consistent with general signature')
            if len(outs) != self._nout:
                raise ValueError(
                    f'signature {sig} for dtypes is invalid number of inputs '
                    'is not consistent with general signature')

            ops.append(_OpsRegister._Op(ins, outs, op))
        return ops

    def _determine_from_args(self, args, casting):
        n = len(args)
        in_types = tuple(arg.dtype for arg in args)
        for op in self._ops:
            op_types = op.in_types
            for i in range(n):
                it = in_types[i]
                ot = op_types[i]
                if not numpy.can_cast(it, ot, casting=casting):
                    break
            else:
                return op
        return None

    def _determine_from_dtype(self, dtype):
        for op in self._ops:
            op_types = op.out_types
            for t in op_types:
                if t != dtype:
                    break
            else:
                return op
        return None

    def _determine_from_signature(self, signature):
        # Lets convert the signature as it can be a tuple of tuples
        # or a string
        if isinstance(signature, tuple):
            # create a string to do a look-up on the ops
            if len(signature) == 1:
                raise TypeError(
                    'The use of a length 1 tuple for the ufunc `signature` is'
                    ' not allowed. Use `dtype` or  fill the tuple with'
                    ' `None`s.')
            nin = self._nin
            nout = self._nout
            if len(signature) != (nin + nout):
                raise TypeError(
                    'A type-tuple must be specified of length 1 or 3 for ufunc'
                    f' {self._name}')
            signature = ''.join(
                numpy.dtype(t).char for t in signature[:nin]) + '->' + ''.join(
                    numpy.dtype(t).char for t in signature[nin:nin+nout])

        if isinstance(signature, str):
            is_out = len(signature) == 1
            for op in self._ops:
                if is_out:
                    for t in op.out_types:
                        if t.char != signature:
                            break
                    else:
                        return op
                else:
                    if op.sig_str == signature:
                        return op
        raise TypeError('No loop matching the specified signature and'
                        f' casting was found for ufunc {self._name}')

    def determine_dtype(self, args, dtype, casting, signature):
        ret_dtype = None
        func = self._default_func
        if signature is not None:
            # TODO(ecastill) use an externally provided signature to
            # find the typecasting rules
            op = self._determine_from_signature(signature)
        elif dtype is not None:
            if type(dtype) == tuple:
                # TODO(ecastill) support dtype tuples
                raise RuntimeError('dtype with tuple is not yet supported')
            op = self._determine_from_dtype(dtype)
        else:
            op = self._determine_from_args(args, casting)

        if op is None:
            # Should we allow op to be none?
            if dtype is None:
                dtype = args[0].dtype
                for arg in args:
                    ret_dtype = numpy.promote_types(dtype, arg.dtype)
            else:
                ret_dtype = get_dtype(dtype)
        else:
            # Convert args to the op specified in_types
            n_args = []
            def argname(): return f'ufunc {self._name} input {i}'
            for i, (arg, in_type) in enumerate(zip(args, op.in_types)):
                _raise_if_invalid_cast(arg.dtype, in_type, casting, argname)

                n_args.append(arg.astype(in_type, copy=False))
            args = n_args
            ret_dtype = op.out_types[0]
            func = op.func

        return args, ret_dtype, func


class _GUFunc:
    '''
    Creates a Generalized Universal Function by wrapping a user
    provided function with the signature.

    ``signature`` determines if the function consumes or produces core
    dimensions. The remaining dimensions in given input arrays (``*args``)
    are considered loop dimensions and are required to broadcast
    naturally against each other.

    Args:
        func (callable):
            Function to call like ``func(*args, **kwargs)`` on input arrays
            (``*args``) that returns an array or tuple of arrays. If
            multiple arguments with non-matching dimensions are supplied,
            this function is expected to vectorize (broadcast) over axes of
            positional arguments in the style of NumPy universal functions.
        signature (string):
            Specifies what core dimensions are consumed and produced by
            ``func``.  According to the specification of numpy.gufunc
            signature.
        supports_batched (bool, optional):
            If the wrapped function supports to pass the complete input
            array with the loop and the core dimensions.
            Defaults to `False`. Dimensions will be iterated in the
            `GUFunc` processing code.
        supports_out (bool, optional):
            If the wrapped function supports out as one of its kwargs.
            Defaults to `False`.
        signatures (list of tuple of str):
            Contains strings in the form of 'ii->i' with i being the char of a
            dtype. Each element of the list is a tuple with the string
            and a alternative function to `func` to be executed when the inputs
            of the function can be casted as described by this function.
        name (str, optional):
            Name for the GUFunc object. If not specified, ``func``'s name
            is used.
        doc (str, optional):
            Docstring for the GUFunc object. If not specified, ``func.__doc__``
            is used.
    '''

    def __init__(self, func, signature, **kwargs):
        # We would like to create gufuncs from cupy regular ufuncs
        # so we can avoid most of the __call__ stuff
        self._func = func
        self._signature = signature
        self.__name__ = kwargs.pop('name', func.__name__)
        self.__doc__ = kwargs.pop('doc', func.__doc__)

        # The following are attributes to avoid applying certain steps
        # when wrapping cupy functions that do some of the gufunc
        # stuff internally due to CUDA libraries requirements
        self._supports_batched = kwargs.pop('supports_batched', False)
        self._supports_out = kwargs.pop('supports_out', False)
        signatures = kwargs.pop('signatures', [])

        if kwargs:
            raise TypeError(
                'got unexpected keyword arguments: '
                + ', '.join([repr(k) for k in kwargs])
            )

        # Preprocess the signature here
        input_coredimss, output_coredimss = _parse_gufunc_signature(
            self._signature)
        self._input_coredimss = input_coredimss
        self._output_coredimss = output_coredimss
        # This is pre-calculated to later check the minimum number of
        # dimensions required per input
        self._min_dims = [0] * len(input_coredimss)
        for i, inp in enumerate(input_coredimss):
            for d in inp:
                if d[-1] != '?':
                    self._min_dims[i] += 1

        # Determine nout: nout = None for functions of one
        # direct return; nout = int for return tuples
        self._nout = (
            0
            if not isinstance(output_coredimss, list)
            else len(output_coredimss)
        )
        self._nin = (
            0
            if not isinstance(input_coredimss, list)
            else len(input_coredimss)
        )
        # Determines the function that will be run depending on the datatypes
        # Pass a list of signatures that are either the types in format
        # ii->o or a tuple with the string and a function other than func to be
        # executed for those types
        # For some reason _nout is a tuple and now we get it with 0s
        self._ops_register = _OpsRegister(
            signatures, self._func, self._nin, self._nout, self.__name__)

    def _apply_func_to_inputs(self, func, dim, sizes, dims, args, outs):
        # Apply function
        # The resulting array is loop_output_dims+the specified dims
        # Some functions have batching logic inside due to higly
        # optimized CUDA libraries so we just call them
        if self._supports_batched or dim == len(dims):
            # Check if the function supports out, order and other args
            if self._supports_out and outs is not None:
                outs = outs[0] if len(outs) == 1 else outs
                func(*args, out=outs)
            else:
                fouts = func(*args)
                # TODO(ecastill) improve this check
                if isinstance(fouts, cupy.ndarray):
                    fouts = (fouts,)
                for o, fo in zip(outs, fouts):
                    cupy._core.elementwise_copy(fo, o)
        else:
            dim_size = sizes[dims[dim]][0]
            for i in range(dim_size):
                n_args = [a[i] for a in args]
                if outs is not None:
                    n_outs = [o[i] for o in outs]
                    self._apply_func_to_inputs(
                        func, dim + 1, sizes, dims, n_args, n_outs)

    def _transpose_element(self, arg, iax, shape):
        iax = tuple(a if a < 0 else a - len(shape) for a in iax)
        tidc = (
            tuple(i for i in range(
                -len(shape) + 0, 0) if i not in iax) + iax
        )
        return arg.transpose(tidc)

    def _get_args_transposed(self, args, input_axes, outs, output_axes):
        # This code credit goes to Dask
        # https://github.com/dask/dask/blob/61b578f5a3ad88cbc6a8b9a73ce08c551bd969fa/dask/array/gufunc.py#L349-L377
        # modifications have been done to support arguments broadcast
        # out argument, and optional core dims.
        transposed_args = []
        # This is used when reshaping the outputs so that we can delete
        # dims that were not specified in the input
        missing_dims = set()
        for i, (arg, iax, input_coredims, md) in enumerate(zip(
                args, input_axes, self._input_coredimss, self._min_dims)):
            shape = arg.shape
            nds = len(shape)
            # For the inputs that has missing dimensions we need to reshape
            if nds < md:
                raise ValueError(f'Input operand {i} does not have enough'
                                 f' dimensions (has {nds}, gufunc core with'
                                 f' signature {self._signature} requires {md}')
            optionals = len(input_coredims) - nds
            if optionals > 0:
                # Look for optional dimensions
                # We only allow the first or the last dimensions to be optional
                if input_coredims[0][-1] == '?':
                    shape = (1,) * optionals + shape
                    missing_dims.update(set(input_coredims[:optionals]))
                else:
                    shape = shape + (1,) * optionals
                    missing_dims.update(
                        set(input_coredims[min(0, len(shape)-1):]))
                arg = arg.reshape(shape)
            transposed_args.append(self._transpose_element(arg, iax, shape))
        args = transposed_args

        if outs is not None:
            transposed_outs = []
            # outs should be transposed to the intermediate form before
            # copying all results
            for out, iox, coredims in zip(
                    outs, output_axes, self._output_coredimss):
                transposed_outs.append(self._transpose_element(
                    out, iox, out.shape))
            # check that outs has been correctly transposed
            # if the function returns a scalar, outs will be ignored
            if len(transposed_outs) == len(outs):
                outs = transposed_outs

        # we cant directly broadcast arrays together since their core dims
        # might differ. Only the loop dimensions are broadcastable
        shape = internal._broadcast_shapes(
            [a.shape[:-len(self._input_coredimss)] for a in args])
        args = [_manipulation.broadcast_to(
            a, shape + a.shape[-len(self._input_coredimss):]) for a in args]

        # Assess input args for loop dims
        input_shapes = [a.shape for a in args]
        num_loopdims = [
            len(s) - len(cd) for s, cd in zip(
                input_shapes, self._input_coredimss)
        ]
        max_loopdims = max(num_loopdims) if num_loopdims else None
        core_input_shapes = [
            dict(zip(icd, s[n:]))
            for s, n, icd in zip(
                input_shapes, num_loopdims, self._input_coredimss)
        ]
        core_shapes = {}
        for d in core_input_shapes:
            core_shapes.update(d)

        loop_input_dimss = [
            tuple(
                '__loopdim%d__' % d for d in range(
                    max_loopdims - n, max_loopdims)
            )
            for n in num_loopdims
        ]
        input_dimss = [li + c for li, c in zip(
            loop_input_dimss, self._input_coredimss)]

        loop_output_dims = max(loop_input_dimss, key=len, default=())

        # Assess input args for same size and chunk sizes
        # Collect sizes and chunksizes of all dims in all arrays
        dimsizess = {}
        for dims, shape in zip(input_dimss, input_shapes):
            for dim, size in zip(dims, shape):
                dimsizes = dimsizess.get(dim, [])
                dimsizes.append(size)
                dimsizess[dim] = dimsizes

        # Assert correct partitioning, for case:
        for dim, sizes in dimsizess.items():
            if set(sizes).union({1}) != {1, max(sizes)}:
                raise ValueError(
                    f'Dimension {dim} with different lengths in arrays'
                )

        return args, dimsizess, loop_output_dims, outs, missing_dims

    def _determine_order(self, args, order):
        if order.upper() in ('C', 'K'):
            # Order is determined to be C to allocate the out array
            # but we will change the strides of the out array
            # to be K later in __call__
            return 'C'
        elif order.upper() == 'A':
            # order is F if all arrays are strictly F
            order = ('F' if all([a.flags.f_contiguous
                                 and not a.flags.c_contiguous
                                 for a in args]) else 'C')
            return order

        elif order.upper() == 'F':
            return 'F'
        else:
            raise RuntimeError(f'Unknown order {order}')

    def __call__(self, *args, **kwargs):
        '''
        Apply a generalized ufunc.

        Args:
            args: Input arguments. Each of them can be a :class:`cupy.ndarray`
                object or a scalar. The output arguments can be omitted or be
                specified by the ``out`` argument.
            axes (List of tuples of int, optional):
                A list of tuples with indices of axes a generalized ufunc
                should operate on.
                For instance, for a signature of ``'(i,j),(j,k)->(i,k)'``
                appropriate for matrix multiplication, the base elements are
                two-dimensional matrices and these are taken to be stored in
                the two last axes of each argument.  The corresponding
                axes keyword would be ``[(-2, -1), (-2, -1), (-2, -1)]``.
                For simplicity, for generalized ufuncs that operate on
                1-dimensional arrays (vectors), a single integer is accepted
                instead of a single-element tuple, and for generalized ufuncs
                for which all outputs are scalars, the output tuples
                can be omitted.
            axis (int, optional):
                A single axis over which a generalized ufunc should operate.
                This is a short-cut for ufuncs that operate over a single,
                shared core dimension, equivalent to passing in axes with
                entries of (axis,) for each single-core-dimension argument
                and ``()`` for all others.
                For instance, for a signature ``'(i),(i)->()'``, it is
                equivalent to passing in ``axes=[(axis,), (axis,), ()]``.
            keepdims (bool, optional):
                If this is set to True, axes which are reduced over will be
                left in the result as a dimension with size one, so that the
                result will broadcast correctly against the inputs. This
                option can only be used for generalized ufuncs that operate
                on inputs that all have the same number of core dimensions
                and with outputs that have no core dimensions , i.e., with
                signatures like ``'(i),(i)->()'`` or ``'(m,m)->()'``.
                If used, the location of the dimensions in the output can
                be controlled with axes and axis.
            casting (str, optional):
                Provides a policy for what kind of casting is permitted.
                Defaults to ``'same_kind'``
            dtype (dtype, optional):
                Overrides the dtype of the calculation and output arrays.
                Similar to signature.
            signature (str or tuple of dtype, optional):
                Either a data-type, a tuple of data-types, or a special
                signature string indicating the input and output types of a
                ufunc. This argument allows you to provide a specific
                signature for the function to be used if registered in the
                ``signatures`` kwarg of the ``__init__`` method.
                If the loop specified does not exist for the ufunc, then
                a TypeError is raised. Normally, a suitable loop is found
                automatically by comparing the input types with what is
                available and searching for a loop with data-types to
                which all inputs can be cast safely. This keyword argument
                lets you bypass that search and choose a particular loop.
            order (str, optional):
                Specifies the memory layout of the output array. Defaults to
                ``'K'``.``'C'`` means the output should be C-contiguous,
                ``'F'`` means F-contiguous, ``'A'`` means F-contiguous
                if the inputs are F-contiguous and not also not C-contiguous,
                C-contiguous otherwise, and ``'K'`` means to match the element
                ordering of the inputs as closely as possible.
            out (cupy.ndarray): Output array. It outputs to new arrays
                default.

        Returns:
            Output array or a tuple of output arrays.
        '''

        #  This argument cannot be used for generalized ufuncs
        #  as those take non-scalar input.
        # where = kwargs.pop('where', None)

        outs = kwargs.pop('out', None)
        axes = kwargs.pop('axes', None)
        axis = kwargs.pop('axis', None)
        order = kwargs.pop('order', 'K')
        dtype = kwargs.pop('dtype', None)
        keepdims = kwargs.pop('keepdims', False)
        signature = kwargs.pop('signature', None)
        casting = kwargs.pop('casting', 'same_kind')
        if len(kwargs) > 0:
            raise RuntimeError(
                'Unknown kwargs {}'.format(' '.join(kwargs.keys())))

        ret_dtype = None
        func = self._func

        # this will cast the inputs appropiately
        args, ret_dtype, func = self._ops_register.determine_dtype(
            args, dtype, casting, signature)

        if not type(self._signature) == str:
            raise TypeError('`signature` has to be of type string')

        if outs is not None and type(outs) != tuple:
            if isinstance(outs, cupy.ndarray):
                outs = (outs,)
            else:
                raise TypeError('`outs` must be a tuple or `cupy.ndarray`')

        filter_order = self._determine_order(args, order)

        input_coredimss = self._input_coredimss
        output_coredimss = self._output_coredimss
        if outs is not None and type(outs) != tuple:
            raise TypeError('`outs` must be a tuple')
        # Axes
        input_axes, output_axes = _validate_normalize_axes(
            axes, axis, keepdims, input_coredimss, output_coredimss
        )

        if len(input_coredimss) != len(args):
            ValueError(
                'According to `signature`, `func` requires %d arguments,'
                ' but %s given' % (len(input_coredimss), len(args)))

        args, dimsizess, loop_output_dims, outs, m_dims = self._get_args_transposed(  # NOQA
            args, input_axes, outs, output_axes)

        # The output shape varies depending on optional dims or not
        # TODO(ecastill) this only works for one out argument
        out_shape = [dimsizess[od][0] for od in loop_output_dims]
        if self._nout > 0:
            out_shape += [dimsizess[od][0] for od in output_coredimss[0]]
        out_shape = tuple(out_shape)

        if outs is None:
            outs = cupy.empty(out_shape, dtype=ret_dtype, order=filter_order)
            if order == 'K':
                strides = internal._get_strides_for_order_K(
                    outs, ret_dtype, out_shape)
                outs._set_shape_and_strides(out_shape, strides, True, True)
            outs = (outs,)
        else:
            if outs[0].shape != out_shape:
                raise ValueError(f'Invalid shape for out {outs[0].shape}'
                                 f' needs {out_shape}')

            _raise_if_invalid_cast(
                ret_dtype, outs[0].dtype, casting, "out dtype")

        self._apply_func_to_inputs(
            func, 0, dimsizess, loop_output_dims, args, outs)

        # This code credit goes to Dask
        # https://github.com/dask/dask/blob/61b578f5a3ad88cbc6a8b9a73ce08c551bd969fa/dask/array/gufunc.py#L462-L503
        # Treat direct output

        if self._nout == 0:
            output_coredimss = [output_coredimss]

        # Split output
        # tmp might be a tuple of outs
        # we changed the way we apply the function compared to dask
        # we have added support for optional dims
        leaf_arrs = []
        for tmp in outs:
            for i, (ocd, oax) in enumerate(zip(output_coredimss, output_axes)):
                leaf_arr = tmp

                # Axes:
                if keepdims:
                    slices = (len(leaf_arr.shape) * (slice(None),)
                              + len(oax) * (numpy.newaxis,))
                    leaf_arr = leaf_arr[slices]

                tidcs = [None] * len(leaf_arr.shape)
                for i, oa in zip(range(-len(oax), 0), oax):
                    tidcs[oa] = i
                j = 0
                for i in range(len(tidcs)):
                    if tidcs[i] is None:
                        tidcs[i] = j
                        j += 1
                leaf_arr = leaf_arr.transpose(tidcs)
                # Delete the dims that were optionals after the input expansion
                if len(m_dims) > 0:
                    shape = leaf_arr.shape
                    # This line deletes the dimensions that were not present
                    # in the input
                    core_shape = shape[-len(ocd):]
                    core_shape = tuple([
                        d for d, n in zip(core_shape, ocd) if n not in m_dims])
                    shape = shape[:-len(ocd)] + core_shape
                    leaf_arr = leaf_arr.reshape(shape)
                # leaf_arrs.append(leaf_arr.astype(leaf_arr.dtype, order=order))  # NOQA
                leaf_arrs.append(leaf_arr)
        return tuple(leaf_arrs) if self._nout > 1 else leaf_arrs[0]
