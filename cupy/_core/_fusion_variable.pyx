import string

import numpy

from cupy._core import _fusion_interface

from cupy._core._scalar cimport get_typename


cdef class _AbstractDim:
    """An abstracted data structure for a length of dimensions.

    Attributes:
        input_index (int):
            The position of the element in the arguments passed to the
            fused function
        axis (int):
            The index of dimensions
    """

    def __init__(self, int input_index, int axis):
        self.input_index = input_index
        self.axis = axis

    def __hash__(self):
        return hash((self.input_index, self.axis))

    def __eq__(self, object other):
        if isinstance(other, int):
            return False
        return (
            self.input_index == other.input_index
            and self.axis == other.axis
        )


class _MemorySpace:
    """A memory space object.

    Attributes:
        id(int): The serial number of memory space.
        base_serial_number(int): The serial number of the base variable
            which have this memory space.
        is_input(bool): If this is set to ``True``, the memory space is
            already allocated as an input array. If this is set to ``False``,
            the memory space should be allocated before launching the kernel.
        is_output(bool): If this is set to ``True``, the memory space is
            used in the return values.
    """
    def __init__(self, memory_id, base_serial_number):
        assert isinstance(memory_id, int)
        assert isinstance(base_serial_number, int)

        self.id = memory_id
        self.base_serial_number = base_serial_number

        # Initially, these attributes are set to be `False`, but might be
        # updated from outside.
        self.is_input = False
        self.is_output = False

    @property
    def is_inout(self):
        """Returns ``True`` if the memory space is used for inputs or outputs.

        If ``True``, the memory space should not be deallocated just after
        the kernel launch. If ``False``, the memory space is used only for
        temporary value in the fused kernel."""
        return self.is_input or self.is_output


class _TraceVariable:
    """Variable object to trace operations in the target function to be fused.

    Attributes:
        index(_MemorySpace): The memory space the variable uses.
        serial_number(int): The serial number of the variable object.
        dtype(dtype): The dtype of the variable.
        rshape(tuple of int): The real shape of the variable.
        ashape(tuple of _AbstractDim): An abstracted shape of the variable.
        input_index(int or None): If not `None`, this variable is used as
            the `input_index`-th input parameter.
        output_index(int or None): If not `None`, this variable is used as
            the `output_index`-th output parameter.
    """
    def __init__(
            self, memory_space, serial_number, dtype, rshape, ashape,
            input_index, output_index):
        assert isinstance(memory_space, _MemorySpace)
        assert isinstance(serial_number, int)
        assert isinstance(dtype, numpy.dtype)
        assert input_index is None or isinstance(input_index, int)
        assert output_index is None or isinstance(output_index, int)
        assert isinstance(rshape, tuple)
        assert isinstance(ashape, tuple)
        assert len(rshape) == len(ashape)
        for rdim, adim in zip(rshape, ashape):
            assert isinstance(rdim, int)
            assert isinstance(adim, (int, _AbstractDim))

        self.memory = memory_space
        self.serial_number = serial_number
        self.dtype = dtype
        self.rshape = rshape
        self.ashape = ashape
        self.input_index = input_index
        self.output_index = output_index

    @property
    def ndim(self):
        return len(self.ashape)

    @property
    def is_base(self):
        return self.serial_number == self.memory.base_serial_number

    @property
    def is_input(self):
        return self.input_index is not None

    @property
    def is_output(self):
        return self.output_index is not None

    @property
    def var_name(self):
        # The name of varialbe stored in global memory space.
        raise NotImplementedError

    @property
    def lvar_name(self):
        # The name of varialbe stored in registers in each thread.
        raise NotImplementedError

    @property
    def indexer_name(self):
        """The name of CUDA CIndxer variable for the variable.
        """
        # TODO(asi1024): Unify indexer with other variables which have the
        # same shape, for performance improvements.
        return 'ind{}_{}'.format(self.memory.id, self.serial_number)

    def format(self, form, **kwargs):
        """Returns a string following the format taken as an input.
        """
        kwargs = dict([
            (k, get_typename(v) if isinstance(v, numpy.dtype) else v)
            for k, v in kwargs.items()]
        )
        return string.Template(form).substitute(
            type=get_typename(self.dtype),
            var=self.var_name,
            lvar=self.lvar_name,
            indexer=self.indexer_name,
            **kwargs
        )

    def __hash__(self):
        assert False, (
            '__hash__ is not defined. Use _VariableSet instead of '
            'set/dict because they do not guarantee the order of contents.')


class _TraceScalar(_TraceVariable):
    """An abstracted scalar object.

    Attributes:
        const_value(scalar object or None): A compile-time constant value.
            Actually, it is `None` iff self.is_input is `True`.
    """

    # TODO(asi1024): Remove index argument.
    def __init__(
            self, index, serial_number, dtype, input_index=None, *,
            const_value=None,):
        super().__init__(
            index, serial_number, dtype, (), (), input_index, None)

        self.const_value = const_value

    @property
    def var_name(self):
        if self.const_value is None:
            return 'a{}'.format(self.memory.id)
        if self.dtype == '?':
            return str(self.const_value).lower()
        if self.dtype.kind == 'c':
            return '{}({}, {})'.format(
                get_typename(self.dtype),
                self.const_value.real,
                self.const_value.imag)
        return str(self.const_value)

    @property
    def lvar_name(self):
        return 'v{}'.format(self.memory.id)

    def as_interface(self):
        return _fusion_interface._ScalarProxy(self)

    def key(self):
        return (self.memory.id,)


class _TraceArray(_TraceVariable):
    """An abstracted array object.

    Attributes:
        broadcasted_from(_TraceArray optional): TODO
        rotated_from(_TraceArray optional): TODO
        axis(int optional): The axis to rotate.
        indexed_from(_TraceArray optional): TODO
        index_key(slice): TODO
    """

    def __init__(
            self, index, serial_number, dtype, input_index=None,
            output_index=None, *, rshape, ashape, **kwargs):

        if ashape is None:
            assert input_index is not None
            ndim = len(rshape)
            ashape = tuple([
                _AbstractDim(input_index, axis) for axis in range(ndim)])

        super().__init__(
            index, serial_number, dtype, rshape, ashape,
            input_index, output_index)

        self._view_of = None
        self.is_broadcast = False
        self.rotate_axis = None
        self.slice_key = None

        if 'broadcasted_from' in kwargs:
            self._view_of = kwargs.pop('broadcasted_from')
            self.is_broadcast = True
        elif 'rotated_from' in kwargs:
            self._view_of = kwargs.pop('rotated_from')
            self.rotate_axis = kwargs.pop('axis')
        elif 'indexed_from' in kwargs:
            self._view_of = kwargs.pop('indexed_from')
            self.slice_key = kwargs.pop('index_key')

        assert len(kwargs) == 0, kwargs

    @property
    def var_name(self):
        return 'a{}_{}'.format(self.memory.id, self.serial_number)

    @property
    def lvar_name(self):
        return 'v{}_{}'.format(self.memory.id, self.serial_number)

    def as_interface(self):
        return _fusion_interface._ArrayProxy(self)

    def make_view(self, serial_number, **kwargs):
        rshape = kwargs.pop('rshape', self.rshape)
        ashape = kwargs.pop('ashape', self.ashape)
        return _TraceArray(
            self.memory, serial_number, self.dtype,
            rshape=rshape, ashape=ashape, **kwargs)

    def key(self):
        """Two variables can be identified if they have the same key.
        """
        if isinstance(self.slice_key, tuple):
            slice_key = []
            for s in self.slice_key:
                if isinstance(s, slice):
                    if not (s.start is None
                            and s.stop is None
                            and s.step in (None, 1, -1)):
                        raise NotImplementedError(
                            'Basic slice supports only x[::] and x[::-1].')
                    slice_key.append((s.start, s.stop, s.step))
                else:
                    slice_key.append(s)
            slice_key = tuple(slice_key)
        else:
            slice_key = self.slice_key

        return (
            self.memory.id, self.ashape, self.input_index,
            getattr(self._view_of, 'serial_number', None),
            self.is_broadcast, self.rotate_axis, slice_key,
        )


class _VariableSet:
    """A stable set of variables
    """

    def __init__(self, *args):
        self.contents = []
        for x in args:
            assert isinstance(x, _TraceVariable)
            if x not in self.contents:
                self.contents.append(x)

    def __len__(self):
        return len(self.contents)

    def item(self):
        assert len(self.contents) == 1
        return self.contents[0]

    def add(self, x):
        if x not in self.contents:
            self.contents.append(x)

    def __iadd__(self, other):
        assert isinstance(other, _VariableSet)
        for x in other.contents:
            self.add(x)
        return self

    def __add__(self, other):
        res = _VariableSet(*self.contents)
        res += other
        return res

    def __contains__(self, elem):
        return elem in self.contents

    def __iter__(self):
        return iter(self.contents)

    def __isub__(self, other):
        assert isinstance(other, _VariableSet)
        for x in other.contents:
            if x in self.contents:
                self.contents.remove(x)
        return self

    def __sub__(self, other):
        res = _VariableSet(*self.contents)
        res -= other
        return res
