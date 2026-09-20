# Functions from the following NumPy document
# https://numpy.org/doc/stable/reference/routines.linalg.html

# -----------------------------------------------------------------------------
# Matrix and vector products
# -----------------------------------------------------------------------------
from __future__ import annotations

from cupy.linalg._product import diagonal  # NOQA
from cupy.linalg._product import dot  # NOQA
from cupy.linalg._product import inner  # NOQA
from cupy.linalg._product import kron  # NOQA
from cupy.linalg._product import linalg_cross as cross  # NOQA
from cupy.linalg._product import matmul  # NOQA
from cupy.linalg._product import matrix_power  # NOQA
from cupy.linalg._product import matrix_transpose  # NOQA
from cupy.linalg._product import multi_dot  # NOQA
from cupy.linalg._product import outer  # NOQA
from cupy.linalg._product import tensordot  # NOQA
from cupy.linalg._product import vdot  # NOQA
from cupy.linalg._product import vecdot  # NOQA

# -----------------------------------------------------------------------------
# Decompositions
# -----------------------------------------------------------------------------
from cupy.linalg._decomposition import cholesky  # NOQA
from cupy.linalg._decomposition import qr  # NOQA
from cupy.linalg._decomposition import svd  # NOQA
from cupy.linalg._decomposition import svdvals  # NOQA

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
from cupy.linalg._norms import matrix_norm  # NOQA
from cupy.linalg._norms import norm  # NOQA
from cupy.linalg._norms import cond  # NOQA
from cupy.linalg._norms import det  # NOQA
from cupy.linalg._norms import matrix_rank  # NOQA
from cupy.linalg._norms import slogdet  # NOQA
from cupy.linalg._norms import trace  # NOQA
from cupy.linalg._norms import vector_norm  # NOQA

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
    # Matrix and vector products
    "cross",
    "diagonal",
    "dot",
    "inner",
    "kron",
    "matmul",
    "matrix_power",
    "matrix_transpose",
    "multi_dot",
    "outer",
    "tensordot",
    "vdot",
    "vecdot",
    # Decompositions
    "cholesky",
    "qr",
    "svd",
    "svdvals",
    # Matrix eigenvalues
    "eigh",
    "eig",
    "eigvalsh",
    "eigvals",
    # Norms and other numbers
    "matrix_norm",
    "norm",
    "cond",
    "det",
    "matrix_rank",
    "slogdet",
    "trace",
    "vector_norm",
    # Solving equations and inverting matrices
    "solve",
    "tensorsolve",
    "lstsq",
    "inv",
    "pinv",
    "tensorinv",
    # Exceptions
    "LinAlgError",
]
