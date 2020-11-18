# Functions from the following SciPy document
# https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

# "NOQA" to suppress flake8 warning
from cupyx.scipy.sparse.linalg._norm import norm  # NOQA
from cupyx.scipy.sparse.linalg._solve import lsqr  # NOQA
from cupyx.scipy.sparse.linalg._eigen import eigsh  # NOQA
from cupyx.scipy.sparse.linalg._eigen import svds  # NOQA
from cupyx.scipy.sparse.linalg._iterative import cg  # NOQA
