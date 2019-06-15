"""
Main fallback class.
"""
import types
import cupy as cp
import numpy as np


from cupy.fallback_mode.utils import _call_cupy
from cupy.fallback_mode.utils import _call_numpy


class _RecursiveAttr:
    """
    RecursiveAttr class to catch all attributes corresponding to numpy,
    when user calls fallback_mode. numpy is an instance of this class.
    """

    to_notify = True

    @classmethod
    def notifications(cls):
        """
        Return notification status
        """
        return cls.to_notify

    @classmethod
    def set_notifications(cls, set_to):
        """
        Set notification status to bool set_to.
        """
        cls.to_notify = set_to

        if cls.to_notify:
            print("Notifications are Enabled")
        else:
            print("Notifications are Disabled")

    def __init__(self, numpy_object, cupy_object):

        self._numpy_object = numpy_object
        self._cupy_object = cupy_object

    @property
    def _cupy_module(self):
        if isinstance(self._cupy_object, types.ModuleType):
            return self._cupy_object
        raise TypeError("'{}' is not a module"
                        .format(self._cupy_object.__name__))

    @property
    def _numpy_module(self):
        if isinstance(self._numpy_object, types.ModuleType):
            return self._numpy_object
        raise TypeError("'{}' is not a module"
                        .format(self._numpy_object.__name__))

    def __getattr__(self, attr):
        """
        Catches attributes corresponding to numpy.

        Runs recursively till attribute gets called.
        Or numpy ScalarType is retrieved.

        Args:
            attr (str): Attribute of _RecursiveAttr class object.

        Returns:
            (_RecursiveAttr object, NumPy scalar):
            Returns_RecursiveAttr object with new numpy_object, cupy_object.
            Returns module, scalars if requested.
        """
        # getting attr
        numpy_object = getattr(self._numpy_object, attr, None)
        cupy_object = getattr(self._cupy_object, attr, None)

        # Retrieval of NumPy scalars
        if isinstance(numpy_object, np.ScalarType):
            return numpy_object

        return _RecursiveAttr(numpy_object, cupy_object)

    def __call__(self, *args, **kwargs):
        """
        Gets invoked when last attribute of _RecursiveAttr class gets called.

        Search for attributes from attr_list in cupy.
        If failed, search in numpy.
        If method is found, calls respective library
        Else, raise AttributeError.

        Args:
            args (tuple): Arguments.
            kwargs (dict): Keyword arguments.

        Returns:
            (module, res, ndarray): Returns of call_cupy() or call_numpy
            Raise AttributeError: If cupy_func and numpy_func is not found.
        """
        # Not callable objects
        if not callable(self._numpy_object) and self._numpy_object is not None:
            raise TypeError("'{}' object is not callable"
                            .format(type(self._numpy_object).__name__))

        # Execute cupy method
        if self._cupy_object is not None:
            return _call_cupy(self._cupy_object, args, kwargs)

        # Notify and execute numpy method
        if self._numpy_object is not None:
            if _RecursiveAttr.to_notify:
                print("'{}' not found in cupy, falling back to numpy"
                      .format(self._numpy_object.__name__))
            return _call_numpy(self._numpy_object, args, kwargs)

        raise AttributeError("Attribute neither in cupy nor numpy")


numpy = _RecursiveAttr(np, cp)
