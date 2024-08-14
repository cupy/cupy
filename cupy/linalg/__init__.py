# Functions from the following NumPy document
# https://numpy.org/doc/stable/reference/routines.linalg.html

# -----------------------------------------------------------------------------
# Matrix and vector products
# -----------------------------------------------------------------------------
from cupy.linalg._product import matrix_power

# -----------------------------------------------------------------------------
# Decompositions
# -----------------------------------------------------------------------------
from cupy.linalg._decomposition import cholesky
from cupy.linalg._decomposition import qr
from cupy.linalg._decomposition import svd

# -----------------------------------------------------------------------------
# Matrix eigenvalues
# -----------------------------------------------------------------------------
from cupy.linalg._eigenvalue import eigh
from cupy.linalg._eigenvalue import eigvalsh

# -----------------------------------------------------------------------------
# Norms and other numbers
# -----------------------------------------------------------------------------
from cupy.linalg._norms import norm
from cupy.linalg._norms import det
from cupy.linalg._norms import matrix_rank
from cupy.linalg._norms import slogdet

# -----------------------------------------------------------------------------
# Solving equations and inverting matrices
# -----------------------------------------------------------------------------
from cupy.linalg._solve import solve
from cupy.linalg._solve import tensorsolve
from cupy.linalg._solve import lstsq
from cupy.linalg._solve import inv
from cupy.linalg._solve import pinv
from cupy.linalg._solve import tensorinv

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
from numpy.linalg import LinAlgError


__all__ = ["matrix_power", "cholesky", "qr", "svd", "eigh", "eigvalsh", "norm",
           "det", "matrix_rank", "slogdet", "solve", "tensorsolve", "inv",
           "pinv", "tensorinv", "LinAlgError"]
