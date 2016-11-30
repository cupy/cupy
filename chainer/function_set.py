import numpy
import warnings

from chainer import cuda
from chainer import link


class FunctionSet(link.Chain):

    """Set of links (as "parameterized functions").

    FunctionSet is a subclass of :class:`~chainer.Chain`. Function
    registration is done just by adding an attribute to :class:`object`.

    .. deprecated:: v1.5
       Use :class:`~chainer.Chain` instead.

       .. note::
          FunctionSet was used for manipulation of one or more parameterized
          functions. The concept of parameterized function is gone, and it has
          been replaced by :class:`~chainer.Link` and :class:`~chainer.Chain`.

    """

    def __init__(self, **links):
        super(FunctionSet, self).__init__(**links)
        warnings.warn('FunctionSet is deprecated. Use Chain instead.',
                      DeprecationWarning)

    def __setattr__(self, key, value):
        d = self.__dict__
        if isinstance(value, link.Link):
            # we cannot use add_link here since add_link calls setattr, and we
            # should allow overwriting for backward compatibility
            if value.name is not None:
                raise ValueError(
                    'given link is already registered to another chain by name'
                    ' %s' % value.name)
            if key in d:
                d[key].name = None
                del d[key]
            else:
                d['_children'].append(key)
            value.name = key
        # deal with properties
        prop = getattr(self.__class__, key, None)
        if isinstance(prop, property) and prop.fset is not None:
            prop.fset(self, value)
        else:
            super(FunctionSet, self).__setattr__(key, value)

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
        """Returns an attribute by name.

        Args:
            key (str): Name of the attribute.

        Returns:
            Attribute.

        .. admonition:: Example

           >>> model = chainer.FunctionSet(l1=L.Linear(10, 10),
           ...                             l2=L.Linear(10, 10))
           >>> l1 = model['l1']  # equivalent to l1 = model.l1

        """
        return getattr(self, key)

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

        The order of parameters is consistent with :meth:`parameters` property.

        """
        return tuple(param.data for param in self.params())

    @parameters.setter
    def parameters(self, params):
        assert len(params) == len([_ for _ in self.params()])
        for dst, src in zip(self.params(), params):
            dst.data = src

    @property
    def gradients(self):
        """Tuple of gradient arrays of all registered functions.

        The order of gradients is consistent with :meth:`parameters` property.

        """
        return tuple(param.grad for param in self.params())

    @gradients.setter
    def gradients(self, grads):
        assert len(grads) == len([_ for _ in self.params()])
        for dst, src in zip(self.params(), grads):
            dst.grad = src
