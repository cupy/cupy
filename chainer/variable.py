import heapq
import weakref

import numpy

from chainer import cuda


class Variable(object):

    """Array with a structure to keep track of computation.

    Every variable holds a data array of type either :class:`numpy.ndarray` or
    :class:`cupy.ndarray`.

    A Variable object may be constructed in two ways: by the user or by some
    function. When a variable is created by some function as one of its
    outputs, the variable holds a reference to that function. This reference is
    used in error backpropagation (a.k.a. backprop). It is also used in
    *backward unchaining*. A variable that does not hold a reference to its
    creator is called a *root* variable. A variable is root if it is created by
    the user, or if the reference is deleted by :meth:`unchain_backward`.

    Users can disable this chaining behavior by setting the volatile flag for
    the initial variables. When a function gets volatile variables as its
    inputs, the output variables do not hold references to the function. This
    acts like unchaining on every function application.

    Attributes:
        data: Data array of type either :class:`numpy.ndarray` or
            :class:`cupy.ndarray`.

        grad: Gradient array. It is ``None`` until backprop reaches this
            variable.

        creator: The function who creates this variable. It is ``None`` if the
            variable is not created by any function.

        volatile: Boolean flag. If True, the variable does not keep track of
            any function applications.

    """

    def __init__(self, data, volatile=False):
        """Initializes a variable.

        Args:
            data (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Data array that this variable holds.
            volatile (bool): Volatility flag. If it is True, the variable will
                not keep track of any function applications.

        """
        assert isinstance(data, (numpy.ndarray, cuda.ndarray))
        assert isinstance(volatile, bool)

        self.data = data
        self.rank = 0
        self.volatile = volatile

        self.splitter = weakref.ref(lambda: 0)  # dead ref
        self._grad = None
        self.creator = None

    def __pos__(self):
        return self

    def __len__(self):
        """Returns the number of elements of the data array.

        Returns:
            int: the number of elements of the data array.

        """
        return self.data.size

    @property
    def label(self):
        """Short text that represents the function."""
        if self.data.shape == ():
            return str(self.data.dtype)
        return '%s, %s' % (str(self.data.shape),
                           str(self.data.dtype))

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, g):
        error_msg = '''
This error is occured in two cases. The first case is when the user manually
sets the Variable.grad incorrectly. The second case is when some Function
implementation has a bug. If you do not manually set the Variable.grad in your
script, please report this error to the issue tracker with the stack trace,
the information of your environment, and your script:
https://github.com/pfnet/chainer/issues/new.
'''
        if g is not None:
            if not isinstance(g, type(self.data)):
                raise TypeError('Type of data and grad mismatch: %s != %s%s'
                                % (type(self.data), type(g), error_msg))
            if g.dtype != self.data.dtype:
                raise TypeError('Dtype of data and grad mismatch: %s != %s%s'
                                % (self.data.dtype, g.dtype, error_msg))
            if g.shape != self.data.shape:
                raise ValueError('Shape of data and grad mismatch: %s != %s%s'
                                 % (self.data.shape, g.shape, error_msg))
        self._grad = g

    def set_creator(self, gen_func):
        """Notifies the variable that the given function is its creator.

        Args:
            gen_func (Function): Function object that creates this variable as
                one of its outputs.

        """
        self.creator = gen_func
        self.rank = gen_func.rank + 1

    def backward(self, retain_grad=False):
        """Runs error backpropagation (a.k.a. backprop) from this variable.

        On backprop, :meth:`Function.backward` is called on each
        :class:`Function` object appearing in the backward graph starting from
        this variable. The backward graph is represented by backward references
        from variables to their creators, and from functions to their inputs.
        The backprop stops at all root variables. Some functions set ``None``
        as gradients of some inputs, where further backprop does not take place
        at such input variables.

        This method uses :data:`grad` as the initial error array. User can
        manually set a gradient array before calling this method. If
        :data:`data` contains only one element (i.e., it is scalar) and
        :data:`grad` is None, then this method automatically complements 1.0 as
        the initial error. This is useful on starting backprop from some scalar
        loss value.

        Args:
            retain_grad (bool): If True, the gradient arrays of all
                intermediate variables are kept. Otherwise, :data:`grad` of the
                intermediate variables are set to ``None`` on appropriate
                timing, which may reduce the maximum memory consumption.

                In most cases of training some model, the purpose of backprop
                is to compute gradients of parameters, not of variables, so it
                is recommended to set this flag False.

        """
        if self.creator is None:
            return

        cand_funcs = []
        seen_set = set()

        # Initilize error by 1, if this is a loss variable
        if self.data.size == 1 and self.grad is None:
            with cuda.get_device(self.data) as device:
                if device is cuda.DummyDevice:
                    self.grad = numpy.ones_like(self.data)
                else:
                    self.grad = cuda.cupy.ones_like(self.data)

        def add_cand(cand):
            if cand is not None and cand not in seen_set:
                # Negate since heapq is min-heap
                heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
                seen_set.add(cand)

        add_cand(self.creator)

        while cand_funcs:
            _, _, func = heapq.heappop(cand_funcs)
            outputs = tuple(y() for y in func.outputs)  # access via weak ref

            in_data = tuple(x.data for x in func.inputs)
            out_grad = tuple(None if y is None else y.grad for y in outputs)
            with cuda.get_device(*(in_data + out_grad)):
                gxs = func.backward(in_data, out_grad)
            assert len(gxs) == len(in_data)

            if not retain_grad:
                for y in outputs:
                    if y is not None and y is not self:
                        y.grad = None
            for x, gx in zip(func.inputs, gxs):
                x.grad = gx
                if gx is not None:  # skip if gradient does not flow
                    add_cand(x.creator)

    def unchain_backward(self):
        """Deletes references between variables and functions backward.

        After this method completes, intermediate variables and functions that
        are not referenced from anywhere are deallocated by reference
        count GC. Also this variable itself deletes the reference to its
        creator function, i.e. this variable becomes root in the computation
        graph. It indicates that backprop after unchaining stops at this
        variable. This behavior is useful to implement truncated BPTT.

        """
        cand_funcs = []
        seen_set = set()

        def add_cand(cand):
            if cand is not None and cand not in seen_set:
                cand_funcs.append(cand)
                seen_set.add(cand)

        add_cand(self.creator)

        while cand_funcs:
            func = cand_funcs.pop()
            for var in func.inputs:
                add_cand(var.creator)
            func.unchain()

    def __lt__(self, other):
        raise NotImplementedError()

    def __le__(self, other):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        raise NotImplementedError()

    def __gt__(self, other):
        raise NotImplementedError()

    def __ge__(self, other):
        raise NotImplementedError()

    def __nonzero__(self):
        raise NotImplementedError()

    def __bool__(self):
        raise NotImplementedError()

    def __hash__(self):
        return super(Variable, self).__hash__()

    __array_priority__ = 200
