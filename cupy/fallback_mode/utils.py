"""
Utilities needed for fallback_mode.
"""
from types import ModuleType
from cupy.fallback_mode.data_tranfer import vram2ram
from cupy.fallback_mode.data_tranfer import ram2vram


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
    def get_attr_list_copy(cls):
        """
        Returns copy of attr_list.
        """
        return cls.attr_list.copy()


def get_last_and_rest(attr_list):
    """
    Returns sub-module and function name using attr_list.

    Args:
        attr_list (list): List which is used for keeping track of attributes.

    Returns:
        path (str): Sub-module
        attr_list[-1] (str): Function name
    """
    path = ".".join(attr_list[:-1])
    return path, attr_list[-1]


def join_attrs(attr_list):
    """
    Returns joined attributes in attr_list.
    """
    path = ".".join(attr_list)
    return path


def get_path(lib, sub_module):
    """
    Returns sub-module path for required library.

    Args:
        lib (str): 'cp' (cupy) or 'np' (numpy).

    Returns:
        path (str): sub-module path.
    """
    if sub_module == '':
        path = lib
    else:
        path = lib + '.' + sub_module

    return path


def call_cupy(func, args, kwargs):
    """
    Calls cupy function with *args and **kwargs.

    Args:
        func: A cupy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        cupy module OR result after calling func.
    """
    if isinstance(func, ModuleType):
        return func

    return func(*args, **kwargs)


def call_numpy(func, args, kwargs):
    """
    Calls numpy function with *args and **kwargs.

    Args:
        func: A numpy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        numpy module OR result after calling func.
    """
    if isinstance(func, ModuleType):
        return func

    cpu_args, cpu_kwargs = vram2ram(args, kwargs)
    cpu_res = func(*cpu_args, **cpu_kwargs)
    gpu_res = ram2vram(cpu_res)

    return gpu_res
