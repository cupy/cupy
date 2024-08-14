# Functions from the following NumPy document
# https://numpy.org/doc/stable/reference/routines.linalg.html

# -----------------------------------------------------------------------------
# Matrix and vector products
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
from numpy.linalg import LinAlgError

# -----------------------------------------------------------------------------
# Decompositions
# -----------------------------------------------------------------------------
from cupy.linalg._decomposition import cholesky, qr, svd

# -----------------------------------------------------------------------------
# Matrix eigenvalues
# -----------------------------------------------------------------------------
from cupy.linalg._eigenvalue import eigh, eigvalsh

# -----------------------------------------------------------------------------
# Norms and other numbers
# -----------------------------------------------------------------------------
from cupy.linalg._norms import det, matrix_rank, norm, slogdet
from cupy.linalg._product import matrix_power

# -----------------------------------------------------------------------------
# Solving equations and inverting matrices
# -----------------------------------------------------------------------------
from cupy.linalg._solve import inv, lstsq, pinv, solve, tensorinv, tensorsolve

__all__ = ["matrix_power", "cholesky", "qr", "svd", "eigh", "eigvalsh", "norm",
           "det", "matrix_rank", "slogdet", "solve", "tensorsolve", "inv",
           "pinv", "tensorinv", "LinAlgError"]
