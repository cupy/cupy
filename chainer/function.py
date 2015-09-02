import copy
import os
import weakref

import numpy
import six

from chainer import cuda
from chainer.utils import type_check
from chainer import variable


class Function(object):

    """Function on variables with backpropagation ability.

    All function implementations defined in :mod:`chainer.functions` inherit
    this class.

    The main feature of this class is keeping track of function applications as
    a backward graph. When a function is applied to :class:`Variable` objects,
    the function is copied, and its :meth:`forward` method is called on
    :data:`~Variable.data` fields of input variables, and at the same time it
    chains references from output variables to the function and from the
    function to its inputs.

    .. note::

       Strictly speaking, when a function is applied to some variable, a
       special :class:`Function` object called *splitter* is inserted between
       the variable and the function. The splitter is used to manipulate
       multiple function applications on the same variable, where gradients
       from different backward paths are accumulated at the variable.

    .. note::

       :meth:`__call__` copies the function instance before the forward
       computation and chaining. This enables us to reuse one function object
       for multiple function applications, where the different calls must use
       different references to the function object. Note that the copy is
       shallow, so implementations of :class:`Function` must take care of any
       member attributes shared accross forward and backward computations.

    .. admonition:: Example

       Let ``x`` an instance of :class:`Variable` and ``f`` an instance of
       :class:`Function` taking only one argument. Then a line

       >>> y = f(x)

       computes a new variable ``y`` and creates backward references. Actually,
       backward references are set as per the following diagram::

           x <--- (splitter) <--- x' <--- f' <--- y

       where prime "'" indicates a copy of the original object. If another
       application the function occurs as

       >>> z = f(x)

       then the splitter acts like a branch as the following new diagram::

                               |--- x'  <--- f'  <--- y
           x <--- (splitter) <-+
                               |--- x'' <--- f'' <--- z

       Note that the splitter is implicitly inserted and user does not need to
       take any special care of it; just remember that such branching is
       correctly managed by chainer.

    Every function implementation should provide :meth:`forward_cpu`,
    :meth:`forward_gpu`, :meth:`backward_cpu` and :meth:`backward_gpu`.
    Alternatively, one can provide :meth:`forward` and :meth:`backward` instead
    of separate methods. Backward methods have default implementations that
    just return ``None``, which indicates that the function is non-
    differentiable.

    Function implementations are classified into two types: parameterized ones
    and non-parameterized ones. A parameterized function holds parameter arrays
    and coresponding gradient arrays. Implementation can choose any way to keep
    these arrays, but it is recommended to keep them as attributes to easily
    migrate between CPU and GPU. Parameterized function must provide accessors
    to these arrays called :meth:`parameters` and :meth:`gradients`.

    Attributes:
        inputs: A tuple or list of input variables.
        outputs: A tuple or list of output variables.
        parameter_names: A tuple or list of names of parameter attributes.
            It is set to an empty tuple by default. This attribute is used by
            the default implementation of :meth:`parameters` property to gather
            the collection of parameter arrays. Implementation of parameterized
            function should override this field as an attribute or a property,
            or otherwise it should override :meth:`parameters` property.
        gradient_names: A tuple or list of names of gradient attributes. The
            detail is same as :data:`parameter_names`.
        type_check_enable: When it is ``True``, the function checks types of
            input arguments. Set ``CHAINER_TYPE_CHECK`` environment variable
            ``0`` to disable type check, or set the variable directly in
            your own program.

    """
    parameter_names = ()
    gradient_names = ()
    type_check_enable = int(os.environ.get('CHAINER_TYPE_CHECK', '1')) != 0

    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.rank = None

    def __call__(self, *inputs):
        """Applies forward propagation with chaining backward references.

        Basic behavior is also expressed in documentation of :class:`Function`
        class. This function first copies itself to avoid conflict over
        multiple invocations.

        .. note::

           If the :data:`~Variable.data` attribute of input variables reside on
           GPU device, then, before it calls :meth:`forward` method, the
           appropriate device is selected, so in most cases implementers do
           not need to take care of device selection.

        Args:
            inputs: Tuple of input :class:`Variable` objects. All input
                variables must have same volatile flag.

        Returns:
            One
            :class:`Variable` object or a tuple of multiple
            :class:`Variable` objects.

        """
        # First copy itself to avoid duplication within the graph.
        self = copy.copy(self)

        if any(x.volatile for x in inputs):  # not build graph
            # do not mix multiple volatility
            assert all(x.volatile for x in inputs)

            in_data = tuple(x.data for x in inputs)
            if self.type_check_enable:
                self._check_data_type_forward(in_data)
            with cuda.get_device(*in_data):
                out_data = self.forward(in_data)
            assert type(out_data) == tuple

            outputs = list(variable.Variable(y, volatile=True)
                           for y in out_data)
            if len(outputs) == 1:
                return outputs[0]
            return outputs

        # Build graph
        # Be careful that forward references must be weak
        self.inputs = []
        for x in inputs:
            splitter = x.splitter()
            if splitter is None:
                splitter = Split(x)
                x.splitter = weakref.ref(splitter)
            self.inputs.append(splitter.add_branch())

        if self.inputs:
            self.rank = max(x.rank for x in self.inputs)
        else:
            self.rank = 0

        in_data = tuple(x.data for x in self.inputs)
        if self.type_check_enable:
            self._check_data_type_forward(in_data)
        with cuda.get_device(*in_data):
            outputs = self.forward(in_data)
        assert type(outputs) == tuple

        ret = tuple(variable.Variable(y) for y in outputs)
        for y in ret:
            y.set_creator(self)

        # Make forward references weak
        self.outputs = tuple(weakref.ref(y) for y in ret)

        if len(ret) == 1:
            return ret[0]
        return ret

    @property
    def label(self):
        """Short text that represents the function.

        The default implementation returns its type name.
        Each function should override it to give more information.
        """
        return self.__class__.__name__

    def _check_data_type_forward(self, in_data):
        in_type = type_check.get_types(in_data, 'in_types', False)
        self.check_type_forward(in_type)

    def check_type_forward(self, in_types):
        """Checks types of input data before forward propagation.

        Before :meth:`forward` is called, this function is called.
        You need to validate types of input data in this function
        using :ref:`the type checking utilities <type-check-utils>`.

        Args:
            in_types (~chainer.utils.type_check.TypeInfoTuple): The type
                information of input data for :meth:`forward`.
        """
        pass

    def forward(self, inputs):
        """Applies forward propagation to input arrays.

        It delegates the procedure to :meth:`forward_cpu` or
        :meth:`forward_gpu` by default. Which it selects is determined by the
        type of input arrays.
        Implementations of :class:`Function` must implement either cpu/gpu
        methods or this method.

        Args:
            inputs: Tuple of input array(s).

        Returns:
            Tuple of output array(s).

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        if any(isinstance(x, cuda.ndarray) for x in inputs):
            return self.forward_gpu(inputs)
        else:
            return self.forward_cpu(inputs)

    def forward_cpu(self, inputs):
        """Applies forward propagation to input arrays on CPU.

        Args:
            inputs: Tuple of :class:`numpy.ndarray` object(s).

        Returns:
            tuple: Tuple of :class:`numpy.ndarray` object(s).

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        raise NotImplementedError()

    def forward_gpu(self, inputs):
        """Applies forward propagation to input arrays on GPU.

        Args:
            inputs: Tuple of :class:`cupy.ndarray` object(s).

        Returns:
            tuple: Tuple of :class:`cupy.ndarray` object(s).

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        raise NotImplementedError()

    def backward(self, inputs, grad_outputs):
        """Applies backprop to output gradient arrays.

        It delegates the procedure to :meth:`backward_cpu` or
        :meth:`backward_gpu` by default. Which it selects is determined by the
        type of input arrays and output gradient arrays. Implementations of
        :class:`Function` must implement either cpu/gpu methods or this method,
        if the function is intended to be backprop-ed.

        Args:
            inputs: Tuple of input arrays.
            grad_outputs: Tuple of output gradient arrays.

        Returns:
            tuple: Tuple of input gradient arrays. Some or all of them can be
            ``None``, if the function is not differentiable on
            inputs.

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        if any(isinstance(x, cuda.ndarray) for x in inputs + grad_outputs):
            return self.backward_gpu(inputs, grad_outputs)
        else:
            return self.backward_cpu(inputs, grad_outputs)

    def backward_cpu(self, inputs, grad_outputs):
        """Applies backprop to output gradient arrays on CPU.

        Args:
            inputs: Tuple of input :class:`numpy.ndarray` object(s).
            grad_outputs: Tuple of output gradient :class:`numpy.ndarray`
                object(s).

        Returns:
            tuple: Tuple of input gradient :class:`numpy.ndarray` object(s).
            Some or all of them can be ``None``, if the function is not
            differentiable on corresponding inputs.

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        return tuple(None for _ in inputs)

    def backward_gpu(self, inputs, grad_outputs):
        """Applies backprop to output gradient arrays on GPU.

        Args:
            inputs: Tuple of input :class:`cupy.ndarray`
                object(s).
            grad_outputs: Tuple of output gradient
                :class:`cupy.ndarray` object(s).

        Returns:
            tuple: Tuple of input gradient :class:`cupy.ndarray`
            object(s). Some or all of them can be ``None``, if the function is
            not differentiable on corresponding inputs.

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        return tuple(None for _ in inputs)

    def unchain(self):
        """Purges in/out variables and this function itself from the graph.

        This method is called from :meth:`Variable.unchain_backward` method.

        """
        for y in self.outputs:
            y_ref = y()
            if y_ref is not None:
                y_ref.creator = None
        for x in self.inputs:
            x.splitter = weakref.ref(lambda: 0)  # dead ref
        self.inputs = None

    def to_gpu(self, device=None):
        """Migrates the function to GPU and returns self.

        The default implementation moves all fields of type
        :class:`numpy.ndarray` onto GPU.

        Args:
            device (int or :class:`cupy.cuda.Device` or ``None``): Device
                ID of GPU that the function will be migrated on. If this is
                ``None``, the current device is used.

        Returns:
            self.

        """
        with cuda.get_device(device):
            for k, v in six.iteritems(self.__dict__):
                if isinstance(v, numpy.ndarray):
                    setattr(self, k, cuda.cupy.array(v))
        return self

    def to_cpu(self):
        """Migrates the function to CPU and returns self.

        The default implementation moves all fields of type
        :class:`cupy.ndarray` onto CPU.

        Returns:
            self.

        """
        for k, v in six.iteritems(self.__dict__):
            if isinstance(v, cuda.ndarray):
                setattr(self, k, v.get())
        return self

    @property
    def parameters(self):
        """A tuple of parameter arrays.

        Default implementation collects parameter arrays based on
        :data:`parameter_names` attribute.

        """
        return tuple(getattr(self, name) for name in self.parameter_names)

    @parameters.setter
    def parameters(self, values):
        assert len(self.parameter_names) == len(values)
        for name, value in zip(self.parameter_names, values):
            setattr(self, name, value)

    @property
    def gradients(self):
        """A tuple of gradient arrays.

        Default implementation collects gradient arrays based on
        :data:`gradient_names` attribute.

        """
        return tuple(getattr(self, name) for name in self.gradient_names)

    @gradients.setter
    def gradients(self, values):
        assert len(self.gradient_names) == len(values)
        for name, value in zip(self.gradient_names, values):
            setattr(self, name, value)


class Split(Function):

    """Special function to branch the graph at variable node.

    Split does not implement forward: it is intended to implicitly used by
    Function.

    """

    def __init__(self, var):
        self.inputs = [var]
        self.outputs = []
        self.rank = var.rank

    def add_branch(self):
        x = self.inputs[0]
        output = variable.Variable(x.data)
        output.set_creator(self)
        self.outputs.append(weakref.ref(output))
        return output

    def backward(self, inputs, grad_outputs):
        # Accumulate gradients
        if len(grad_outputs) == 1:
            return grad_outputs  # no copy

        gx = None
        grad_outputs = [gy for gy in grad_outputs if gy is not None]
        with cuda.get_device(*grad_outputs):
            for gy in grad_outputs:
                if gx is None:
                    gx = gy.copy()
                else:
                    gx += gy

        return gx,
