"""
Main fallback class.
"""
import cupy as cp # NOQA
import numpy as np # NOQA


from cupy.fallback_mode.utils import FallbackUtil
from cupy.fallback_mode.utils import get_last_and_rest
from cupy.fallback_mode.utils import join_attrs
from cupy.fallback_mode.utils import call_cupy
from cupy.fallback_mode.utils import call_numpy


class Recursive_attr:
    """
    Recursive_attr class to catch all attributes corresponding to numpy,
    when user calls fallback_mode. numpy is an instance of this class.
    """
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        """
        Catches and appends attributes corresponding to numpy
        to attr_list.
        Runs recursively till attribute gets called by returning
        dummy object of this class.

        Args:
            attr (str): Attribute of Recursive_attr class object.

        Returns:
            dummy : Recursive_attr object.
        """
        if self.name == 'numpy':
            FallbackUtil.clear_attrs()
        FallbackUtil.add_attrs(attr)
        return dummy

    def __call__(self, *args, **kwargs):
        """
        Gets invoked when last dummy attributes gets called.
        Transfers call to fallback object along with *args, **kwargs.
        """
        return fallback(*args, **kwargs)


class Fallback(FallbackUtil):
    """
    Fallback class where fallback_mode is going to be executed after
    getting list of attributes.
    """
    def __call__(self, *args, **kwargs):
        """
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
        attributes = FallbackUtil.get_attr_list_copy()
        sub_module, func_name = get_last_and_rest(attributes)

        # trying cupy
        try:
            if sub_module == '':
                cupy_path = 'cp'
            else:
                cupy_path = 'cp' + '.' + sub_module

            cupy_func = getattr(eval(cupy_path), func_name)

            return call_cupy(cupy_func, args, kwargs)

        except AttributeError:
            # trying numpy
            if fallback.notifications:
                if sub_module == "":
                    print("Attribute '{}' not found in cupy. falling back to\
                           numpy".format(func_name))
                else:
                    print("Attribute '{}.{}' not found in cupy. falling back\
                           to numpy".format(sub_module, func_name))

            if sub_module == '':
                numpy_path = 'np'
            else:
                numpy_path = 'np' + '.' + sub_module

            numpy_func = getattr(eval(numpy_path), func_name)

            return call_numpy(numpy_func, args, kwargs)

        except AttributeError:
            raise AttributeError("Attribute {} neither in cupy nor numpy"
                                 .format(join_attrs(attributes)))


numpy = Recursive_attr('numpy')
dummy = Recursive_attr('dummy')
fallback = Fallback()
