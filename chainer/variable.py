import collections
import heapq
import traceback
import warnings

import numpy
import six

import chainer
from chainer import cuda
from chainer import flag
from chainer import utils


def _check_grad_type(func, x, gx):
    def make_message(message):
        if func:
            detail = 'Function `{0}` ({1}) has a bug.\n'.format(
                type(func).__name__, func.label)

            stack = func.stack
            if stack:
                detail += 'Stacktrace of the function is below:\n'
                for line in traceback.format_list(func._stack):
                    detail += line

            detail += '''
Please report this error to the issue tracker with the stack trace,
the information of your environment, and your script:
https://github.com/pfnet/chainer/issues/new.
'''.format(type(func).__name__, func.label)

        else:
            detail = ''

        detail += message
        return detail

    if not isinstance(gx, type(x.data)):
        msg = ('Type of data and grad mismatch\n%s != %s' %
               (type(x.data), type(gx)))
        raise TypeError(make_message(msg))
    if gx.dtype != x.data.dtype:
        msg = ('Dtype of data and grad mismatch\n%s != %s' %
               (x.data.dtype, gx.dtype))
        raise TypeError(make_message(msg))
    if gx.shape != x.data.shape:
        msg = ('Shape of data and grad mismatch\n%s != %s' %
               (x.data.shape, gx.shape))
        raise ValueError(make_message(msg))


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

    Args:
        data (array): Initial data array.
        volatile (~chainer.Flag): Volatility flag. String ('on', 'off', or
            'auto') or boolean values can be used, too.
        name (str): Name of the variable.
        grad (array): Initial gradient array.

    Attributes:
        data: Data array of type either :class:`numpy.ndarray` or
            :class:`cupy.ndarray`.
        grad: Gradient array.
        creator: The function who creates this variable. It is ``None`` if the
            variable is not created by any function.
        volatile: Ternary :class:`~chainer.Flag` object. If ``'ON'``, the
            variable does not keep track of any function applications. See
            :class:`~chainer.Flag` for the detail of ternary flags.

    """

    def __init__(self, data, volatile=flag.OFF, name=None, grad=None):
        if not isinstance(data, (numpy.ndarray, cuda.ndarray)):
            msg = '''numpy.ndarray or cuda.ndarray are expected.
Actual: {0}'''.format(type(data))
            raise TypeError(msg)

        self.data = data
        self.rank = 0
        self._volatile = flag.Flag(volatile)

        self._grad = grad
        self.creator = None

        self.name = name

    def __reduce__(self):
        return Variable, (self.data, self.volatile, self.name, self._grad)

    def __repr__(self):
        if self.name:
            return '<variable %s>' % self.name
        else:
            return '<variable at 0x%x>' % id(self)

    def __str__(self):
        return self.name or ('<var@%x>' % id(self))

    def debug_print(self):
        """Display a summary of the stored data and location of the Variable"""

        msg = """{summary}
- device: {device}
- volatile: {volatile}
- backend: {background}
- shape: {shape}
- dtype: {dtype}
- statistics: {stats}
- grad: {grad}"""

        stats_msg = 'mean={0:.8f}, std={1:.8f}'

        try:
            device = self.data.device
        except AttributeError:
            device = 'CPU'

        with cuda.get_device(self.data) as dev:
            xp = numpy if int(dev) == -1 else cuda.cupy

            if self.grad is None:
                grad = None
            elif xp.all(self.grad == 0):
                grad = 0
            else:
                grad = stats_msg.format(float(xp.mean(self.grad)),
                                        float(xp.std(self.grad)))

            stats = stats_msg.format(float(xp.mean(self.data)),
                                     float(xp.std(self.data)))

        return msg.format(summary=repr(self), volatile=self.volatile,
                          grad=grad, shape=self.data.shape,
                          background=type(self.data),
                          dtype=self.data.dtype, device=device,
                          stats=stats)

    def __pos__(self):
        return self

    def __len__(self):
        """Returns the number of elements of the data array.

        Returns:
            int: Number of elements of the data array.

        """
        return self.data.size

    @property
    def volatile(self):
        return self._volatile

    @volatile.setter
    def volatile(self, v):
        self._volatile = flag.Flag(v)

    @property
    def label(self):
        """Short text that represents the variable."""
        if self.data.shape == ():
            return str(self.data.dtype)
        return '(%s), %s' % (', '.join(map(str, self.data.shape)),
                             str(self.data.dtype))

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, g):
        if g is not None:
            _check_grad_type(None, self, g)
        self._grad = g

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def to_cpu(self):
        """Copies the data and gradient arrays to CPU."""
        self.data = cuda.to_cpu(self.data)
        if self._grad is not None:
            self._grad = cuda.to_cpu(self._grad)

    def to_gpu(self, device=None):
        """Copies the data and gradient arrays to specified GPU.

        Args:
            device: Target device specifier. If omitted, the current device is
                used.

        """
        with cuda.get_device(device):
            self.data = cuda.to_gpu(self.data)
            if self._grad is not None:
                self._grad = cuda.to_gpu(self._grad)

    def cleargrad(self):
        """Clears the gradient array."""
        self._grad = None

    def zerograd(self):
        """Initializes the gradient array by zeros.

        .. deprecated:: v1.15
           Use :meth:`cleargrad` instead.

        """
        warnings.warn(
            'Variable.zerograd is deprecated. Use Variable.cleargard instead.',
            DeprecationWarning)
        with cuda.get_device(self.data) as dev:
            if self._grad is None:
                xp = numpy if int(dev) == -1 else cuda.cupy
                self._grad = xp.zeros_like(self.data)
            else:
                self._grad.fill(0)

    def copydata(self, var):
        """Copies the data array from given source variable.

        This method just copies the data attribute from given variable to this
        variable, except that the copy is even done across the host and
        different devices.

        Args:
            var (Variable): Source variable.

        """
        src = var.data
        dst = self.data
        src_xp = cuda.get_array_module(src)
        dst_xp = cuda.get_array_module(dst)
        if dst_xp is src_xp:
            dst_xp.copyto(dst, src)
        elif dst_xp is numpy:
            dst_xp.copyto(dst, src.get())
        else:
            dst.set(src)

    def addgrad(self, var):
        """Accumulates the gradient array from given source variable.

        This method just runs ``self.grad += var.grad``, except that the
        accumulation is even done across the host and different devices.

        Args:
            var (Variable): Source variable.

        """
        src = var._grad
        dst = self._grad
        if src is None:
            return

        src_dev = cuda.get_device(src)
        dst_dev = cuda.get_device(self.data)

        if src_dev.id == dst_dev.id:
            with dst_dev:
                if dst is None:
                    xp = cuda.get_array_module(src)
                    self._grad = xp.copy(src)
                else:
                    self._grad += src
            return

        if dst_dev.id < 0:
            src_grad = cuda.to_cpu(src)
        else:
            src_grad = cuda.to_gpu(src, device=dst_dev)

        if dst is None:
            self._grad = src_grad
        else:
            with dst_dev:
                self._grad += src_grad

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
        :data:`grad` is ``None``, then this method automatically complements
        1.0 as the initial error. This is useful on starting backprop from
        some scalar loss value.

        Args:
            retain_grad (bool): If ``True``, the gradient arrays of all
                intermediate variables are kept. Otherwise, :data:`grad` of the
                intermediate variables are set to ``None`` on appropriate
                timing, which may reduce the maximum memory consumption.

                In most cases of training some models, the purpose of backprop
                is to compute gradients of parameters, not of variables, so it
                is recommended to set this flag ``False``.

        """
        if self.creator is None:
            return
        initial_device = None
        if cuda.available:
            try:
                initial_device = cuda.Device()
            except cuda.cupy.cuda.runtime.CUDARuntimeError as e:
                if e.status != 38:  # cudaErrorNoDevice
                    raise

        is_debug = chainer.is_debug()

        cand_funcs = []
        seen_set = set()
        seen_vars = set()
        need_copy = set()

        # Initialize error by 1, if this is a loss variable
        if self.data.size == 1 and self.grad is None:
            with cuda.get_device(self.data) as device:
                if device is cuda.DummyDevice:
                    self.grad = numpy.ones_like(self.data)
                else:
                    self.grad = cuda.cupy.ones_like(self.data)

        def add_cand(cand):
            if cand not in seen_set:
                # Negate since heapq is min-heap
                heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
                seen_set.add(cand)

        add_cand(self.creator)

        while cand_funcs:
            _, _, func = heapq.heappop(cand_funcs)
            outputs = [y() for y in func.outputs]  # access via weak ref

            in_data = tuple([x.data for x in func.inputs])
            out_grad = tuple([None if y is None else y.grad for y in outputs])
            hooks = chainer.get_function_hooks()
            if func._n_local_function_hooks != 0:
                hooks = collections.OrderedDict(hooks)
                hooks.update(func.local_function_hooks)

            cuda.get_device(*(in_data + out_grad)).use()
            for hook in six.itervalues(hooks):
                hook.backward_preprocess(func, in_data, out_grad)
            gxs = func.backward(in_data, out_grad)
            assert len(gxs) == len(in_data)
            for hook in six.itervalues(hooks):
                hook.backward_postprocess(func, in_data, out_grad)

            if is_debug:
                for gx in gxs:
                    if gx is None:
                        continue
                    cuda.get_device(gx).use()
                    if cuda.get_array_module(gx).isnan(gx).any():
                        msg = 'NaN is detected on backward computation'
                        raise RuntimeError(msg)

            if not retain_grad:
                for y in outputs:
                    if y is not None and y is not self:
                        y.grad = None
            for x, gx in zip(func.inputs, gxs):
                if gx is None:
                    continue

                _check_grad_type(func, x, gx)

                # Accumulate the gradient to x. It is a bit tricky to handle
                # branches and parameter gradient accumulation correctly.
                id_x = id(x)
                if x.creator is None:  # leaf
                    if x._grad is None:
                        x.grad = gx
                        need_copy.add(id_x)
                    else:
                        cuda.get_device(gx).use()
                        if id_x in need_copy:
                            x.grad = utils.force_array(x.grad + gx)  # copy
                            need_copy.remove(id_x)
                        else:
                            x._grad += gx
                else:  # not a leaf
                    add_cand(x.creator)
                    if id_x not in seen_vars:  # 1st visit
                        x.grad = gx
                        seen_vars.add(id_x)
                        need_copy.add(id_x)
                    else:
                        cuda.get_device(gx).use()
                        if id_x in need_copy:  # 2nd visit
                            x._grad = utils.force_array(gx + x._grad)  # copied
                            need_copy.remove(id_x)
                        else:  # 3rd or later visit
                            x._grad += gx
            del gxs  # to reduce memory usage
            if initial_device is not None:
                initial_device.use()

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
