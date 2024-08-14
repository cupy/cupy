# Functions from the following SciPy document
# https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

from cupyx.scipy.sparse.linalg._norm import norm
from cupyx.scipy.sparse.linalg._solve import spsolve
from cupyx.scipy.sparse.linalg._solve import spsolve_triangular
from cupyx.scipy.sparse.linalg._solve import factorized
from cupyx.scipy.sparse.linalg._solve import lsqr
from cupyx.scipy.sparse.linalg._solve import lsmr
from cupyx.scipy.sparse.linalg._solve import splu
from cupyx.scipy.sparse.linalg._solve import spilu
from cupyx.scipy.sparse.linalg._solve import SuperLU
from cupyx.scipy.sparse.linalg._solve import minres
from cupyx.scipy.sparse.linalg._eigen import eigsh
from cupyx.scipy.sparse.linalg._eigen import svds
from cupyx.scipy.sparse.linalg._iterative import cg
from cupyx.scipy.sparse.linalg._iterative import gmres
from cupyx.scipy.sparse.linalg._iterative import cgs
from cupyx.scipy.sparse.linalg._interface import LinearOperator
from cupyx.scipy.sparse.linalg._interface import aslinearoperator
from cupyx.scipy.sparse.linalg._lobpcg import lobpcg
