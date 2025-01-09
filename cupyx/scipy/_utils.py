import functools

import numpy
import scipy._lib


def enable_scipy_array_api(scipy_func, required_version):

    def decorator(cupy_func):
        @functools.wraps(cupy_func)
        def wrapper(*args, **kwargs):
            if numpy.lib.NumpyVersion(scipy.__version__) >= required_version:
                config = scipy._lib._array_api._GLOBAL_CONFIG
                old = config['SCIPY_ARRAY_API']
                config['SCIPY_ARRAY_API'] = True
                out = scipy_func(*args, **kwargs)
                config['SCIPY_ARRAY_API'] = old
                return out
            return cupy_func(*args, **kwargs)
        return wrapper

    return decorator
