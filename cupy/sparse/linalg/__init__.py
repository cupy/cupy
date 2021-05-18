import sys
import warnings

import cupyx.scipy.sparse.linalg


if (3, 7) <= sys.version_info:
    def __getattr__(name):
        if hasattr(cupyx.scipy.sparse.linalg, name):
            warnings.warn(
                'cupy.sparse.linalg is deprecated.'
                ' Use cupyx.scipy.sparse.linalg instead.', DeprecationWarning)
            return getattr(cupyx.scipy.sparse.linalg, name)
        raise AttributeError(
            "module 'cupy.sparse.linalg' has no attribute {!r}".format(name))
else:
    from cupyx.scipy.sparse.linalg import *  # NOQA
