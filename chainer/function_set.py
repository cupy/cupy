import numpy
import six
import warnings

from chainer import cuda
from chainer import function


class FunctionSet(object):

    """Set of objects with ``parameters`` and ``gradients`` properties.

    :class:`FunctionSet` is useful to collect parameters and gradients of
    multiple parameterized :class:`Function` objects. :class:`FunctionSet`
    itself also implements :attr:`~FunctionSet.parameters` and
    :attr:`~FunctionSet.gradients`, so it can be nested in another
    :class:`FunctionSet` object.

    Function registration is done by just adding an attribute to
    :class:`FunctionSet` object.

    """

    def __init__(self, **functions):
        """Initializes the function set by given functions.

        Args:
            **functions: ``dict`` of ``str`` key and :class:`Function` values.
                The key-value pairs are just set to the :class:`FunctionSet`
                object as attributes.

        """
        for name, func in six.iteritems(functions):
            setattr(self, name, func)

    def collect_parameters(self):
        """Returns a tuple of parameters and gradients.

        Returns:
            Tuple (pair) of two tuples. The first element is a tuple of
            parameter arrays, and the second is a tuple of gradient arrays.

        """

        msg = ("'collect_parameters' is deprecated. "
               "You can pass FunctionSet itself to 'optimizer.setup'")
        warnings.warn(msg, FutureWarning)
        return self

    def __getitem__(self, key):
        """Returns the :class:`Function` objects by name.

        Args:
            key (str): Name of the function.

        Returns:
            ~chainer.Function: Function object.

        .. admonition:: Example

           >>> model = FunctionSet(l1=F.Linear(10, 10), l2=F.Linear(10, 10))
           >>> l1 = model['l1']
        """

        return getattr(self, key)

    def to_gpu(self, device=None):
        """Migrates all parameters and gradients onto GPU.

        This method calls ``to_gpu`` method of each registered object.

        Args:
            device (int or :class:`cupy.cuda.Device` or ``None``): Device
                ID of GPU. If ``None`` is given, it uses the current device.

        Returns:
            self

        """
        for func in six.itervalues(self.__dict__):
            if isinstance(func, (function.Function, FunctionSet)):
                func.to_gpu(device=device)
        return self

    def to_cpu(self):
        """Migrates all parameters and gradients onto CPU.

        This method calls ``to_cpu`` method of each registered object.

        Returns:
            self

        """
        for func in six.itervalues(self.__dict__):
            if isinstance(func, (function.Function, FunctionSet)):
                func.to_cpu()
        return self

    def copy_parameters_from(self, params):
        """Copies parameters from another source without reallocation.

        Args:
            params (Iterable): Iterable of parameter arrays.

        """
        for dst, src in zip(self.parameters, params):
            if isinstance(dst, numpy.ndarray):
                if isinstance(src, numpy.ndarray):
                    numpy.copyto(dst, src)
                else:
                    dst[:] = src.get()
            elif isinstance(src, numpy.ndarray):
                dst.set(src)
            else:
                cuda.copy(src, out=dst)

    @property
    def parameters(self):
        """Tuple of parameter arrays of all registered functions.

        The order of parameters is consistent with :meth:`gradients` property.

        """
        return sum((func.parameters for _, func in self._get_sorted_funcs()),
                   ())

    @parameters.setter
    def parameters(self, params):
        param_iter = iter(params)
        for _, func in self._get_sorted_funcs():
            func.parameters = param_iter

    @property
    def gradients(self):
        """Tuple of gradient arrays of all registered functions.

        The order of gradients is consistent with :meth:`parameters` property.

        """
        return sum((func.gradients for _, func in self._get_sorted_funcs()),
                   ())

    @gradients.setter
    def gradients(self, grads):
        grad_iter = iter(grads)
        for _, func in self._get_sorted_funcs():
            func.gradients = grad_iter

    def _get_sorted_funcs(self):
        return sorted(
            [func_tuple for func_tuple in six.iteritems(self.__dict__)
             if isinstance(func_tuple[1], (function.Function, FunctionSet))])
