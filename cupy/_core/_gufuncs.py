from __future__ import annotations

import re

import numpy

import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core.internal import _normalize_axis_indices
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal


# Signature parsing code and dimension accessing has been borrowed
# from dask
# https://github.com/dask/dask/blob/61b578f5a3ad88cbc6a8b9a73ce08c551bd969fa/dask/array/gufunc.py#L12-L55
_DIMENSION_NAME = r'\w+(\?|\|1)?'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*,?)?'.format(_DIMENSION_NAME)
_ARGUMENT = r'\({}\)'.format(_CORE_DIMENSION_LIST)
_INPUT_ARGUMENTS = '(?:{0:}(?:,{0:})*,?)?'.format(_ARGUMENT)
_OUTPUT_ARGUMENTS = '{0:}(?:,{0:})*'.format(
    _ARGUMENT
)  # Use `'{0:}(?:,{0:})*,?'` if gufunc-
# signature should be allowed for length 1 tuple returns
_SIGNATURE = '^{:}->{:}$'.format(_INPUT_ARGUMENTS, _OUTPUT_ARGUMENTS)


def _parse_gufunc_signature(signature):
    """Parse and prepare gufunc signature for internal use, we convert
    each operand into a tuple of dimensions and for each dimension return
    (name, optional, broadcastable).
    Additionally, returns the number of unique core dimensions (for later).
    """
    coredims = set()

    # The code has been modified from dask to support optional dimensions
    if not isinstance(signature, str):
        raise TypeError('Signature is not a string')

    if signature == '' or signature is None:
        raise ValueError('Signature cannot be empty')

    signature = signature.replace(' ', '')
    if not re.match(_SIGNATURE, signature):
        raise ValueError('Not a valid gufunc signature: {}'.format(signature))
    in_txt, out_txt = signature.split('->')
    ins = [list(x.split(',')) if x != '' else ()
           for x in in_txt[1:-1].split('),(')]
    outs = [list(y.split(',')) if y != '' else ()
            for y in out_txt[1:-1].split('),(')]

    # Modify the signature information to be a tuple of
    # (name, optional, broadcastable)
    for cds in ins:
        for i in range(len(cds)):
            optional = cds[i].endswith('?')
            broadcastable = cds[i].endswith('|1')
            # Remove either (can't be both)
            name = cds[i].removesuffix('?').removesuffix('|1')
            cds[i] = (name, optional, broadcastable)

    for cds in outs:
        for i in range(len(cds)):
            optional = cds[i].endswith('?')
            broadcastable = cds[i].endswith('|1')
            if broadcastable:
                raise ValueError('Output name cannot indicate |1.')
            # Remove either (can't be both)
            name = cds[i].removesuffix('?')
            cds[i] = (name, optional, broadcastable)

    ins = [tuple(_) for _ in ins]
    outs = [tuple(_) for _ in outs]

    for cds in ins + outs:
        for cd in cds:
            coredims.add(cd[0])  # add coredim name

    return ins, outs, len(coredims)


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
        if self._nout != 1:
            # dtype discovery doesn't yet support this (the rest should)
            raise NotImplementedError('Multiple output gufuncs not supported')
        ret_dtype = None
        func = self._default_func
        if signature is not None:
            # TODO(ecastill) use an externally provided signature to
            # find the typecasting rules
            op = self._determine_from_signature(signature)
        elif dtype is not None:
            if isinstance(dtype, tuple):
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

    def __init__(
            self, func, signature, *,
            name=None, doc=None,
            supports_batched=False, supports_out=False, signatures=None):
        # We would like to create gufuncs from cupy regular ufuncs
        # so we can avoid most of the __call__ stuff
        self._func = func
        self._signature = signature
        self.__name__ = name if name is not None else func.__name__
        self.__doc__ = doc if doc is not None else func.__doc__

        # The following are attributes to avoid applying certain steps
        # when wrapping cupy functions that do some of the gufunc
        # stuff internally due to CUDA libraries requirements
        self._supports_batched = supports_batched
        self._supports_out = supports_out
        signatures = signatures if signatures is not None else []

        # Preprocess the signature here
        input_coredimss, output_coredimss, cndim_ix = _parse_gufunc_signature(
            self._signature)
        self._core_num_dim_ix = cndim_ix
        self._core_dims = input_coredimss + output_coredimss
        self._input_coredimss = input_coredimss
        self._output_coredimss = output_coredimss
        # This is pre-calculated to later check the minimum number of
        # dimensions required per input
        self._min_dims = [0] * len(input_coredimss)
        for i, inp in enumerate(input_coredimss):
            for d in inp:
                if d[1]:  # name, optional, broadcastable
                    self._min_dims[i] += 1

        self._nout = len(output_coredimss)
        self._nin = len(input_coredimss)

        # Determines the function that will be run depending on the datatypes
        # Pass a list of signatures that are either the types in format
        # ii->o or a tuple with the string and a function other than func to be
        # executed for those types
        # For some reason _nout is a tuple and now we get it with 0s
        self._ops_register = _OpsRegister(
            signatures, self._func, self._nin, self._nout, self.__name__)

    def _validate_normalize_axes(self, arrays, axes, axis, keepdims):
        # This code originally came from Dask and was later heavily modified:
        # https://github.com/dask/dask/blob/61b578f5a3ad88cbc6a8b9a73ce08c551bd969fa/dask/array/gufunc.py#L58-L172
        if axes is not None and axis is not None:
            raise ValueError(
                'Only one of `axis` or `axes` keyword arguments '
                'should be given')
        if axes and not isinstance(axes, list):
            raise ValueError('`axes` has to be of type list')

        if keepdims:
            # Store the number of core dimensions to append into keepdims.
            keepdims = None
            for icd in self._input_coredimss:
                ndim_core = len(icd)
                if keepdims is not None and keepdims != ndim_core:
                    raise ValueError(
                        '`keepdims` requires all inputs to have the same '
                        'number of core dimensions')
                keepdims = ndim_core

            for ocd in self._output_coredimss:
                if len(ocd) > 0:
                    raise ValueError(
                        '`keepdims` can only be used for scalar outputs')

        if axis is not None:
            if not isinstance(axis, int):
                raise ValueError('`axis` argument has to be an integer value')
            if self._core_num_dim_ix != 1:
                raise ValueError(
                    '`axis` can be used only, if there is a single shared '
                    f'core dimension, but {self._core_num_dim_ix} are '
                    f'present in the signature {self._signature}'
                )
            if not keepdims:
                axes = [(axis,) if cd else () for cd in self._core_dims]
            else:
                # When keepdims is used with axis, it applies to the output
                axes = [(axis,) if cd else () for cd in self._input_coredimss]
                axes.extend([(axis,)] * self._nout)
        elif axes is not None:
            # Allow a single integer in axes.
            axes = [(a,) if isinstance(a, int) else a for a in axes]
        else:
            axes = [None] * (self._nin + self._nout)

        if len(axes) == self._nin:
            for ocd in self._output_coredimss:
                if len(ocd) > 0:
                    raise ValueError(
                        '`axes` entries for outputs can only be omitted if '
                        'none of them has core axes.')
            # Output axes may be ommitted, fill them in if this is the case.
            # This seems actually more relaxes as NumPy as of NumPy 2.4.
            axes.extend([None] * self._nout)
        elif len(axes) != self._nin + self._nout:
            raise ValueError(
                'The number of `axes` entries is not equal to the number of '
                'input and output arguments. Outputs may only be omitted if '
                'they are all scalar.')

        return axes[:self._nin], axes[self._nin:], keepdims

    def _get_transpose(self, i, axs, ndim, n_core_dims):
        """Normalize the axes tuple and check the bounds may be None at this
        point.
        """
        if axs is None:
            if ndim < n_core_dims:
                raise ValueError(
                    f'Argument {i} has too few dimensions.')
            return None
        elif len(axs) != n_core_dims:
            # NOTE(seberg): Like NumPy, this rejects partial axes when axes
            # are optional. The kernel (e.g. matmul) would need the info.
            raise cupy.exceptions.AxisError(
                f'Number of axes passed for argument {i} does not match the '
                f'the number of core dimensions.')

        axs = _normalize_axis_indices(axs, ndim, sort_axes=False)
        if axs == tuple(range(ndim-n_core_dims, ndim)):
            return None  # normalize for no transpose needed

        return tuple(i for i in range(ndim) if i not in axs) + axs

    def _apply_func_to_inputs(self, func, outer_shape, args, outs):
        # Apply function
        # The resulting array is loop_output_dims+the specified dims
        # Some functions have batching logic inside due to highly
        # optimized CUDA libraries so we just call them
        if self._supports_batched or len(outer_shape) == 0:
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
            # iterate over first outer dimension and recurse.
            for i in range(outer_shape[0]):
                n_args = [a[i] for a in args]
                n_outs = [o[i] for o in outs]
                self._apply_func_to_inputs(
                    func, outer_shape[1:], n_args, n_outs)

    def _update_dims(self, i, core_dims, cd, length):
        # Helper to update dimensions, just to not repeat error messages.
        name = cd[0]
        prev_length = core_dims.get(name, None)
        if prev_length is None:
            if cd[2] and length == 1:
                return  # broadcastable, other operand may set it.
            core_dims[name] = length
        elif length == 1 and cd[2]:  # cd[2] indicates broadcastable
            # broadcastable core-dim of size 1, ignore it.
            return
        elif prev_length != length:
            if prev_length == -1 or length == -1:
                raise ValueError(
                    'An optional core-dimension must be skipped in '
                    'all or no inputs.')
            raise ValueError(
                f'Input operand {i} has mismatch in its core '
                f'dimension {name} with signature {self._signature} '
                f'({prev_length} != {length}).')

    def _setup_operands(
            self, args, in_axes, outs, out_axes, keepdims,
            ret_dtype, filter_order, casting):
        """Set up the operands for the function call, this needs to figure out
        the actual core axes and shapes and make sure core-dimensions match.
        We then transpose the core dimensions to the end for the actual
        operation.

        This function also sets up the output operands, note that there are two
        versions. The untransposed ones for user-return and the transposed ones
        for the actual result.
        """
        core_dims = {}
        outer_shapes = []
        transposals = []
        for i, (arg, axs, coredims) in enumerate(
                zip(args, in_axes, self._input_coredimss)):
            ndim = arg.ndim
            n_core_dims = max(self._min_dims[i], min(ndim, len(coredims)))
            transpose = self._get_transpose(i, axs, ndim, n_core_dims)
            if transpose is None:
                outer_shape = arg.shape[:-n_core_dims]
                core_shape = arg.shape[-n_core_dims:]
            else:
                outer_shape = tuple(arg.shape[i]
                                    for i in transpose[:-n_core_dims])
                core_shape = tuple(arg.shape[i]
                                   for i in transpose[-n_core_dims:])

            n_skipped = len(coredims) - n_core_dims
            ommitted_coredims = []
            i_coredim = 0
            for length in core_shape:
                # Process the core dimension shape and store the result into
                # `core_dims` for each dimension.

                # If there are optional coredims skip them from the front
                while (cd := coredims[i_coredim])[1] and n_skipped > 0:
                    ommitted_coredims.append(i_coredim)
                    self._update_dims(i, core_dims, cd, -1)
                    i_coredim += 1
                    n_skipped -= 1
                else:
                    i_coredim += 1

                self._update_dims(i, core_dims, cd, length)

            # The above may not have processed all optional dimensions, do it
            # to ensure errors for missing ones (and simplify output shape).
            while i_coredim < len(coredims):
                ommitted_coredims.append(i_coredim)
                self._update_dims(i, core_dims, coredims[i_coredim], -1)
                i_coredim += 1

            if ommitted_coredims:
                args[i] = cupy.expand_dims(arg, axis=tuple(ommitted_coredims))

            outer_shapes.append(outer_shape)
            transposals.append(transpose)

        # The outer shape needs to be broadcast across all input operands.
        bc_outer_shape = internal._broadcast_shapes(outer_shapes)

        for i, (arg, outer_shape) in enumerate(zip(args, outer_shapes)):
            # Above, we figured out the transpose and outer shape, we still
            # need to apply it (potentially)
            transpose = transposals[i]
            if transpose is not None:
                arg = arg.transpose(transpose)
            if outer_shape != bc_outer_shape:
                arg = _manipulation.broadcast_to(
                    arg, bc_outer_shape + arg.shape[len(outer_shape):])

            args[i] = arg

        untransposed_outs = []  # The return array is the untransposed one.
        for i, (out, axs, coredims) in enumerate(
                zip(outs, out_axes, self._output_coredimss)):
            ommitted_dims = []
            if not keepdims:
                core_shape = []
                for i_cd, cd in enumerate(coredims):
                    dim = core_dims.get(cd[0], None)
                    if dim == -1:
                        ommitted_dims.append(i_cd + len(bc_outer_shape))
                    elif dim is None:
                        core_shape.append(1)  # should be a |1 core-dim
                    else:
                        core_shape.append(dim)
            else:
                assert not self._output_coredimss[i]
                core_shape = [1] * keepdims

            shape = bc_outer_shape + tuple(core_shape)
            trans = self._get_transpose(
                self._nin + i, axs, len(shape), len(core_shape))

            if out is not None:
                untransposed_outs.append(out)
                if trans is not None:
                    out = out.transpose(trans)
                if out.shape != shape:
                    # Inverse transpose for error message (if needed)
                    itrans = range(len(shape))
                    if trans is not None:
                        itrans = sorted(itrans, key=lambda i: trans[i])
                    actual = tuple(out.shape[i] for i in itrans)
                    expected = tuple(shape[i] for i in itrans)
                    raise ValueError(
                        f'Output operand {i} has invalid shape {actual} '
                        f'expected {expected}')

                _raise_if_invalid_cast(
                    ret_dtype, out.dtype, casting, "out dtype")
            else:
                if trans is not None:
                    itrans = sorted(range(len(trans)), key=lambda i: trans[i])
                    shape = tuple(shape[i] for i in itrans)
                # Note: Order logic here may be weird/wrong as core dims should
                # be preferred contiguous.
                out = cupy.empty(shape, dtype=ret_dtype, order=filter_order)
                untransposed_outs.append(out)
                if trans is not None:
                    out = out.transpose(trans)

            if ommitted_dims:
                out = cupy.expand_dims(out, axis=tuple(ommitted_dims))
            if keepdims:
                # The core function should not see the keepdims, strip them.
                out = out[(...,) + (0,) * keepdims]
            outs[i] = out  # The operand "out" may have been transposed

        return args, outs, untransposed_outs, bc_outer_shape

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

    def __call__(
            self, *args,
            axes=None, axis=None, keepdims=False, casting='same_kind',
            dtype=None, signature=None, order='K', out=None):
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
        ret_dtype = None
        func = self._func

        # this will cast the inputs appropriately
        args, ret_dtype, func = self._ops_register.determine_dtype(
            args, dtype, casting, signature)
        args = list(args)  # make args mutable to transpose later
        if len(args) != self._nin:
            raise ValueError(
                'According to `signature`, `func` requires %d arguments,'
                ' but %s given' % (self._nin, len(args)))

        if not isinstance(self._signature, str):
            raise TypeError('`signature` has to be of type string')

        if out is None:
            outs = [None] * self._nout
        elif not isinstance(out, tuple):
            if not isinstance(out, cupy.ndarray):
                raise TypeError('`out` must be a tuple or `cupy.ndarray`')
            outs = [out]
        else:
            outs = list(out)  # make outs mutable to transpose later
            for out in outs:
                if out is not None and not isinstance(out, cupy.ndarray):
                    raise TypeError(
                        '`out` tuple must contain `cupy.ndarray` or None')

        filter_order = self._determine_order(args, order)

        # Preproces axes/axis (does not check out of bound axes)
        in_axes, out_axes, keepdims = self._validate_normalize_axes(
            args, axes, axis, keepdims)

        args, outs, untransposed_outs, outer_shape = self._setup_operands(
            args, in_axes, outs, out_axes, keepdims, ret_dtype, filter_order,
            casting)

        self._apply_func_to_inputs(func, outer_shape, args, outs)

        if len(untransposed_outs) == 1:
            return untransposed_outs[0]
        else:
            return tuple(untransposed_outs)
