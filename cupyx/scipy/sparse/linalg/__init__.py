# Functions from the following SciPy document
# https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

# "NOQA" to suppress flake8 warning
from cupyx.scipy.sparse.linalg._norm import norm  # NOQA
from cupyx.scipy.sparse.linalg._solve import spsolve  # NOQA
from cupyx.scipy.sparse.linalg._solve import spsolve_triangular  # NOQA
from cupyx.scipy.sparse.linalg._solve import factorized  # NOQA
from cupyx.scipy.sparse.linalg._solve import lsqr  # NOQA
from cupyx.scipy.sparse.linalg._solve import splu  # NOQA
from cupyx.scipy.sparse.linalg._solve import spilu  # NOQA
from cupyx.scipy.sparse.linalg._solve import SuperLU  # NOQA
from cupyx.scipy.sparse.linalg._eigen import eigsh  # NOQA
from cupyx.scipy.sparse.linalg._eigen import svds  # NOQA
from cupyx.scipy.sparse.linalg._iterative import cg  # NOQA
from cupyx.scipy.sparse.linalg._iterative import gmres  # NOQA
from cupyx.scipy.sparse.linalg._interface import LinearOperator  # NOQA
from cupyx.scipy.sparse.linalg._interface import aslinearoperator  # NOQA
from cupyx.scipy.sparse.linalg._lobpcg import lobpcg  # NOQA
