"""
Main fallback class.

TODO: dispatching function for execution
"""
import cupy as cp # NOQA
import numpy as np # NOQA
from types import ModuleType

from utils import FallbackUtil
from utils import get_last_and_rest
from utils import join_attrs


class Recursive_attr:

    def __getattr__(self, attr):
        FallbackUtil.add_attrs(attr)
        return self

    def __call__(self, *args, **kwargs):
        return now_fallback(*args, **kwargs)


class Fallback(FallbackUtil):

    def __call__(self, *args, **kwargs):
        attributes = FallbackUtil.get_attr_list_copy()
        FallbackUtil.clear_attrs()

        sub_module, func_name = get_last_and_rest(attributes)

        # trying cupy
        try:
            if sub_module == '':
                cupy_path = 'cp'
            else:
                cupy_path = 'cp' + '.' + sub_module
            cupy_func = getattr(eval(cupy_path), func_name)

            # call_cupy() will be called here when Implemented
            if isinstance(cupy_func, ModuleType):
                return cupy_func
            print("fallback to be applied on '{}' which is in '{}' with arguments:\n{}\n{}"
                  .format(cupy_func.__name__, cupy_func.__module__, args, kwargs))

        except AttributeError:
            # trying numpy
            if FallbackUtil.notifications:
                if sub_module == "":
                    print("no attribute '{}' found in cupy. Falling back to numpy"
                          .format(func_name))
                else:
                    print("no attribute '{}.{}' found in cupy. Falling back to numpy"
                          .format(sub_module, func_name))
            if sub_module == '':
                numpy_path = 'np'
            else:
                numpy_path = 'np' + '.' + sub_module
            numpy_func = getattr(eval(numpy_path), func_name)

            # call_numpy() will be called here when Implemented
            if isinstance(numpy_func, ModuleType):
                return numpy_func
            print("fallback to be applied on '{}' which is in '{}' with arguments:\n{}\n{}"
                  .format(numpy_func.__name__, numpy_func.__module__, args, kwargs))

        except AttributeError:
            raise AttributeError("{} neither in cupy nor numpy"
                                 .format(join_attrs(attributes)))

        print("other steps")


numpy = Recursive_attr()
now_fallback = Fallback()
