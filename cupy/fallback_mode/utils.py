"""
Utilities needed for fallback_mode.
"""
from types import ModuleType
from cupy.fallback_mode.data_tranfer import vram2ram
from cupy.fallback_mode.data_tranfer import ram2vram


class FallbackUtil:

    attr_list = []

    def __init__(self):
        self.notifications = True

    def notification_status(self):
        return self.notifications

    def set_notification_status(self, status):
        self.notifications = status
        print("Notification status is now {}".format(self.notifications))

    @classmethod
    def clear_attrs(cls):
        cls.attr_list = []

    @classmethod
    def add_attrs(cls, attr):
        cls.attr_list.append(attr)

    @classmethod
    def get_attr_list_copy(cls):
        return cls.attr_list.copy()


def get_last_and_rest(attr_list):
    path = ".".join(attr_list[:-1])
    return path, attr_list[-1]


def join_attrs(attr_list):
    path = ".".join(attr_list)
    return path


def call_cupy(func, args, kwargs):

    if isinstance(func, ModuleType):
        return func

    return func(*args, **kwargs)


def call_numpy(func, args, kwargs):

    if isinstance(func, ModuleType):
        return func

    cpu_args, cpu_kwargs = vram2ram(args, kwargs)

    cpu_res = func(*cpu_args, **cpu_kwargs)

    gpu_res = ram2vram(cpu_res)

    return gpu_res
