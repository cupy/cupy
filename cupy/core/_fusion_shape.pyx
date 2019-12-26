from cupy.core import _fusion_thread_local


cdef class _AbstractDim(object):
    """An abstrated data structure for length of dimensions.
    """

    cdef:
        readonly int input_index
        readonly int axis

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


cdef class _ShapeConstraints(object):
    """The data structure which manages the conditions between the shapes.
    """

    cdef:
        readonly list eq_constraints
        readonly list const_constraints

    def __init__(self):
        self.eq_constraints = []
        self.const_constraints = []

    def add_eq_constraint(self, x, y):
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
        _fusion_thread_local.check_not_runtime()
        assert isinstance(x, (_AbstractDim, int))
        assert isinstance(value, int)
        x = self.evaluate(x)
        if isinstance(x, _AbstractDim):
            self.const_constraints.append((x, value))
        else:
            assert x == value

    def evaluate(self, x):
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
        for a, b in self.eq_constraints:
            if dim_map[a] != dim_map[b]:
                return False
        for a, b in self.const_constraints:
            if dim_map[a] != b:
                return False
        return True
