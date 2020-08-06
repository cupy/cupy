# Functions from the following NumPy document
# https://docs.scipy.org/doc/numpy/reference/routines.linalg.html

# -----------------------------------------------------------------------------
# Matrix and vector products
# -----------------------------------------------------------------------------
from cupy.linalg.product import matrix_power  # NOQA

# -----------------------------------------------------------------------------
# Decompositions
# -----------------------------------------------------------------------------
from cupy.linalg.decomposition import cholesky  # NOQA
from cupy.linalg.decomposition import qr  # NOQA
from cupy.linalg.decomposition import svd  # NOQA

# -----------------------------------------------------------------------------
# Matrix eigenvalues
# -----------------------------------------------------------------------------
from cupy.linalg.eigenvalue import eigh  # NOQA
from cupy.linalg.eigenvalue import eigvalsh  # NOQA

# -----------------------------------------------------------------------------
# Norms and other numbers
# -----------------------------------------------------------------------------
from cupy.linalg.norms import norm  # NOQA
from cupy.linalg.norms import det  # NOQA
from cupy.linalg.norms import matrix_rank  # NOQA
from cupy.linalg.norms import slogdet  # NOQA

# -----------------------------------------------------------------------------
# Solving equations and inverting matrices
# -----------------------------------------------------------------------------
from cupy.linalg.solve import solve  # NOQA
from cupy.linalg.solve import tensorsolve  # NOQA
from cupy.linalg.solve import lstsq  # NOQA
from cupy.linalg.solve import inv  # NOQA
from cupy.linalg.solve import pinv  # NOQA
from cupy.linalg.solve import tensorinv  # NOQA
