import sys
import warnings

import cupyx.scipy.sparse


# Raise a `DeprecationWarning` for `cupy.sparse` submodule when its functions
# are called. We could raise the warning on importing the submodule, but we
# use module level `__getattr__` function here as the submodule is also
# imported in cupy/__init__.py. Unfortunately, module level `__getattr__` is
# supported on Python 3.7 and higher, so we need to keep the explicit import
# list for older Python.
if (3, 7) <= sys.version_info:
    def __getattr__(name):
        if hasattr(cupyx.scipy.sparse, name):
            msg = 'cupy.sparse is deprecated. Use cupyx.scipy.sparse instead.'
            warnings.warn(msg, DeprecationWarning)
            return getattr(cupyx.scipy.sparse, name)
        raise AttributeError(
            "module 'cupy.sparse' has no attribute {!r}".format(name))
else:
    from cupyx.scipy.sparse import *  # NOQA
