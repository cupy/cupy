# Functions from the following NumPy document
# https://docs.scipy.org/doc/numpy/reference/routines.linalg.html

# -----------------------------------------------------------------------------
# Matrix and vector products
# -----------------------------------------------------------------------------
from cupy.linalg._product import matrix_power  # NOQA

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
from cupy.linalg._eigenvalue import eigvalsh  # NOQA

# -----------------------------------------------------------------------------
# Norms and other numbers
# -----------------------------------------------------------------------------
from cupy.linalg._norms import norm  # NOQA
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
