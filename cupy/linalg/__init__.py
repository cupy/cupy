# Functions from the following NumPy document
# https://numpy.org/doc/stable/reference/routines.linalg.html

# -----------------------------------------------------------------------------
# Matrix and vector products
# -----------------------------------------------------------------------------
from cupy.linalg._product import matrix_power  # NOQA
from cupy.linalg._product import linalg_cross as cross  # NOQA

# -----------------------------------------------------------------------------
# Decompositions
# -----------------------------------------------------------------------------
from cupy.linalg._decomposition import cholesky  # NOQA
from cupy.linalg._decomposition import qr  # NOQA
from cupy.linalg._decomposition import svd  # NOQA

# -----------------------------------------------------------------------------
# Matrix eigenvalues
# -----------------------------------------------------------------------------
from cupy.linalg._eigenvalue import eigh  # NOQA
from cupy.linalg._eigenvalue import eig  # NOQA
from cupy.linalg._eigenvalue import eigvalsh  # NOQA
from cupy.linalg._eigenvalue import eigvals  # NOQA

# -----------------------------------------------------------------------------
# Norms and other numbers
# -----------------------------------------------------------------------------
from cupy.linalg._norms import norm  # NOQA
from cupy.linalg._norms import cond  # NOQA
from cupy.linalg._norms import det  # NOQA
from cupy.linalg._norms import matrix_rank  # NOQA
from cupy.linalg._norms import slogdet  # NOQA

# -----------------------------------------------------------------------------
# Solving equations and inverting matrices
# -----------------------------------------------------------------------------
from cupy.linalg._solve import solve  # NOQA
from cupy.linalg._solve import tensorsolve  # NOQA
from cupy.linalg._solve import lstsq  # NOQA
from cupy.linalg._solve import inv  # NOQA
from cupy.linalg._solve import pinv  # NOQA
from cupy.linalg._solve import tensorinv  # NOQA

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
from numpy.linalg import LinAlgError  # NOQA


__all__ = [
    "matrix_power",
    "cholesky",
    "qr",
    "svd",
    "eigh",
    "eig",
    "eigvalsh",
    "eigvals",
    "norm",
    "cond",
    "det",
    "matrix_rank",
    "slogdet",
    "solve",
    "tensorsolve",
    "inv",
    "pinv",
    "tensorinv",
    "LinAlgError",
]
