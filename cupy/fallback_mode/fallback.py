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

    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        if self.name == 'numpy':
            FallbackUtil.clear_attrs()
        FallbackUtil.add_attrs(attr)
        return dummy

    def __call__(self, *args, **kwargs):
        return fallback(*args, **kwargs)


class Fallback(FallbackUtil):

    def __call__(self, *args, **kwargs):
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
                    print("Attribute '{}' not found in cupy. falling back to numpy"
                          .format(func_name))
                else:
                    print("Attribute '{}.{}' not found in cupy. falling back to numpy"
                          .format(sub_module, func_name))
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
