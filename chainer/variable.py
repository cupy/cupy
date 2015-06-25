import heapq
import weakref

import numpy

from chainer import cuda


class Variable(object):

    """Array with a structure to keep track of computation.

    Every variable holds a data array of type either :class:`~numpy.ndarray` or
    :class:`~pycuda.gpuarray.GPUArray`.

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
        data: Data array of type either :class:`~numpy.ndarray` or
            :class:`~pycuda.gpuarray.GPUArray`.

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
            data (:class:`~numpy.ndarray` or \
                :class:`~pycuda.gpuarray.GPUArray`):
                Data array that this variable holds.
            volatile (bool): Volatility flag. If it is True, the variable will
                not keep track of any function applications.

        .. warning::

           If the data array is of type :class:`~pycuda.gpuarray.GPUArray`, its
           allocator must be :func:`chainer.cuda.mem_alloc`, which allocates
           device memory using device-wise memory pool. All functions of
           :mod:`cuda` automatically uses this allocator.

        """
        self.data = data
        self.rank = 0
        self.volatile = volatile

        self.splitter = weakref.ref(lambda: 0)  # dead ref
        self.grad = None
        self.creator = None

    def __pos__(self):
        return self

    def __len__(self):
        """Returns the number of elements of the data array.

        Returns:
            int: the number of elements of the data array.

        """
        return self.data.size

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
        :data:`grad` is None, then this method automatically complement 1.0 as
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
            with cuda.using_device(self.data) as user:
                if user.is_active:
                    self.grad = cuda.ones_like(self.data)
                else:
                    self.grad = numpy.ones_like(self.data)

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
            out_grad = tuple(y and y.grad for y in outputs)
            with cuda.using_device(*(in_data + out_grad)):
                gxs = func.backward(in_data, out_grad)
            assert len(gxs) == len(in_data)

            if not retain_grad:
                for y in outputs:
                    if y is not None and y != self:
                        y.grad = None
            for x, gx in zip(func.inputs, gxs):
                x.grad = gx
                if gx is not None:  # skip if gradient does not flow
                    add_cand(x.creator)

    def unchain_backward(self):
        """Deletes backward references of variables and functions backward,
        a.k.a. *backward unchaining*.

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

    __array_priority__ = 200
