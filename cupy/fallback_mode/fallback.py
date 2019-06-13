"""
Main fallback class.
"""
import types
import cupy as cp
import numpy as np

from cupy.fallback_mode.utils import FallbackUtil
from cupy.fallback_mode.utils import call_cupy
from cupy.fallback_mode.utils import call_numpy


class RecursiveAttr(FallbackUtil):
    """
    RecursiveAttr class to catch all attributes corresponding to numpy,
    when user calls fallback_mode. numpy is an instance of this class.
    """
    def __init__(self, name):
        self.name = name

    def notification_status(self):
        return super().notification_status()

    def set_notification_status(self, status):
        return super().set_notification_status(status)

    def __getattr__(self, attr):
        """
        Catches and appends attributes corresponding to numpy
        to attr_list.
        Runs recursively till attribute gets called by returning
        dummy object of this class. Or module is requested.

        Args:
            attr (str): Attribute of RecursiveAttr class object.

        Returns:
            (dummy, module, scalars): dummy RecursiveAttr object.
                                      Returns module, scalars if requested.
        """
        # Initialize attr_list
        if self.name == 'numpy':
            super().clear_attrs()

        # direct retrieval of numpy scalars
        if self.name == 'numpy':
            scalar = None
            if hasattr(np, attr):
                scalar = getattr(np, attr)
            if scalar is not None and isinstance(scalar, np.ScalarType):
                return scalar

        # Requesting module
        if attr == '_numpy_module' or attr == '_cupy_module':
            attributes = super().get_attr_list_copy()

            # retrieving cupy module
            if attr == '_cupy_module':
                if self.name == 'numpy':
                    return cp

                func = super().get_func('cp', attributes)

                if isinstance(func, types.ModuleType):
                    return func
                raise TypeError("'{}' is not a module"
                                .format(".".join(attributes)))

            # retrieving numpy module
            if self.name == 'numpy':
                return np

            func = super().get_func('np', attributes)
            if isinstance(func, types.ModuleType):
                return func
            raise TypeError("'{}' is not a module"
                            .format(".".join(attributes)))

        super().add_attrs(attr)
        return dummy

    def __call__(self, *args, **kwargs):
        """
        Gets invoked when last dummy attribute gets called.

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
        attributes = super().get_attr_list_copy()

        # don't call numpy module
        if self.name == 'numpy':
            raise TypeError("'module' object is not callable")

        try:
            # trying cupy
            cupy_func = super().get_func('cp', attributes)

            return call_cupy(cupy_func, args, kwargs)

        except AttributeError:

            try:
                # trying numpy
                if super().notification_status():

                    sub_module = ".".join(attributes[:-1])
                    func_name = attributes[-1]

                    if sub_module == "":
                        print("'{}' not found in cupy, falling back to numpy"
                              .format(func_name))
                    else:
                        print("'{}' not found in cupy, falling back to numpy"
                              .format(sub_module + '.' + func_name))

                numpy_func = super().get_func('np', attributes)

                return call_numpy(numpy_func, args, kwargs)

            except AttributeError:
                raise AttributeError("Attribute '{}' neither in cupy nor numpy"
                                     .format(".".join(attributes)))


numpy = RecursiveAttr('numpy')
dummy = RecursiveAttr('dummy')
