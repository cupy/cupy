# Functions from the following SciPy document
# https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

from cupyx.scipy.sparse.linalg._eigen import eigsh, svds
from cupyx.scipy.sparse.linalg._interface import (
    LinearOperator,
    aslinearoperator,
)
from cupyx.scipy.sparse.linalg._iterative import cg, cgs, gmres
from cupyx.scipy.sparse.linalg._lobpcg import lobpcg
from cupyx.scipy.sparse.linalg._norm import norm
from cupyx.scipy.sparse.linalg._solve import (
    SuperLU,
    factorized,
    lsmr,
    lsqr,
    minres,
    spilu,
    splu,
    spsolve,
    spsolve_triangular,
)
