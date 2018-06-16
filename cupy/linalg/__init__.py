# Functions from the following NumPy document
# https://docs.scipy.org/doc/numpy/reference/routines.linalg.html

# "NOQA" to suppress flake8 warning
from cupy.linalg import decomposition  # NOQA
from cupy.linalg import eigenvalue  # NOQA
from cupy.linalg import einsum  # NOQA
from cupy.linalg import norms  # NOQA
from cupy.linalg.norms import det  # NOQA
from cupy.linalg.norms import matrix_rank  # NOQA
from cupy.linalg.norms import norm  # NOQA
from cupy.linalg.norms import slogdet  # NOQA
from cupy.linalg import product  # NOQA
from cupy.linalg import solve  # NOQA

from cupy.linalg.decomposition import cholesky  # NOQA
from cupy.linalg.decomposition import qr  # NOQA
from cupy.linalg.decomposition import svd  # NOQA

from cupy.linalg.eigenvalue import eigh  # NOQA
from cupy.linalg.eigenvalue import eigvalsh  # NOQA

from cupy.linalg.solve import inv  # NOQA
from cupy.linalg.solve import pinv  # NOQA
from cupy.linalg.solve import solve  # NOQA
from cupy.linalg.solve import tensorinv  # NOQA
from cupy.linalg.solve import tensorsolve  # NOQA

from cupy.linalg.product import matrix_power  # NOQA
