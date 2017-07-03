# Functions from the following NumPy document
# http://docs.scipy.org/doc/numpy/reference/routines.linalg.html

# "NOQA" to suppress flake8 warning
from cupy.linalg import decomposition  # NOQA
from cupy.linalg import eigenvalue  # NOQA
from cupy.linalg import norms  # NOQA
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
