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
            "module 'cupyx.scipy.sparse' has no attribute '{}'".format(name))
else:
    from cupyx.scipy.sparse.base import issparse  # NOQA
    from cupyx.scipy.sparse.base import isspmatrix  # NOQA
    from cupyx.scipy.sparse.base import spmatrix  # NOQA
    from cupyx.scipy.sparse.coo import coo_matrix  # NOQA
    from cupyx.scipy.sparse.coo import isspmatrix_coo  # NOQA
    from cupyx.scipy.sparse.csc import csc_matrix  # NOQA
    from cupyx.scipy.sparse.csc import isspmatrix_csc  # NOQA
    from cupyx.scipy.sparse.csr import csr_matrix  # NOQA
    from cupyx.scipy.sparse.csr import isspmatrix_csr  # NOQA
    from cupyx.scipy.sparse.dia import dia_matrix  # NOQA
    from cupyx.scipy.sparse.dia import isspmatrix_dia  # NOQA

    from cupyx.scipy.sparse.construct import eye  # NOQA
    from cupyx.scipy.sparse.construct import identity  # NOQA
    from cupyx.scipy.sparse.construct import rand  # NOQA
    from cupyx.scipy.sparse.construct import random  # NOQA
    from cupyx.scipy.sparse.construct import spdiags  # NOQA

    from cupyx.scipy.sparse.construct import bmat  # NOQA
    from cupyx.scipy.sparse.construct import hstack  # NOQA
    from cupyx.scipy.sparse.construct import vstack  # NOQA

    # TODO(unno): implement bsr_matrix
    # TODO(unno): implement dok_matrix
    # TODO(unno): implement lil_matrix

    from cupyx.scipy.sparse.construct import kron  # NOQA
    # TODO(unno): implement kronsum
    # TODO(unno): implement diags
    # TODO(unno): implement block_diag
    # TODO(unno): implement tril
    # TODO(unno): implement triu

    # TODO(unno): implement save_npz
    # TODO(unno): implement load_npz

    # TODO(unno): implement find

    # TODO(unno): implement isspmatrix_bsr(x)
    # TODO(unno): implement isspmatrix_lil(x)
    # TODO(unno): implement isspmatrix_dok(x)

    from cupyx.scipy.sparse import linalg  # NOQA
