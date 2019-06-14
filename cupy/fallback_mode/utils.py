"""
Utilities needed for fallback_mode.
"""
import cupy as cp # NOQA
import numpy as np # NOQA

from cupy.fallback_mode.data_tranfer import _get_cupy_result
from cupy.fallback_mode.data_tranfer import _get_numpy_args


class FallbackUtil:
    """
    FallbackUtil class
    Contains utilities needed for fallback.
    """
    attr_list = []
    notifications = True

    @classmethod
    def notification_status(cls):
        """
        Returns notification status.
        """
        return cls.notifications

    @classmethod
    def set_notification_status(cls, status):
        """
        Sets notification status.
        """
        cls.notifications = status
        print("Notification status is now {}".format(cls.notifications))

    @classmethod
    def get_attr_list_copy(cls):
        return cls.attr_list.copy()

    @classmethod
    def clear_attrs(cls):
        """
        Initialize empty attr_list.
        """
        cls.attr_list = []

    @classmethod
    def add_attrs(cls, attr):
        """
        Add attribute name to attr_list.
        """
        cls.attr_list.append(attr)

    @classmethod
    def get_func(cls, lib, attr_list):

        sub_module = ".".join(attr_list[:-1])
        func_name = attr_list[-1]

        if sub_module == '':
            path = lib
        else:
            path = lib + '.' + sub_module

        func = getattr(eval(path), func_name)

        return func


def call_cupy(func, args, kwargs):
    """
    Calls cupy function with *args and **kwargs.

    Args:
        func: A cupy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func.
    """

    return func(*args, **kwargs)


def call_numpy(func, args, kwargs):
    """
    Calls numpy function with *args and **kwargs.

    Args:
        func: A numpy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func.
    """

    numpy_args, numpy_kwargs = _get_numpy_args(args, kwargs)
    numpy_res = func(*numpy_args, **numpy_kwargs)
    cupy_res = _get_cupy_result(numpy_res)

    return cupy_res
