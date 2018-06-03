# Functions from the following SciPy document
# https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

# "NOQA" to suppress flake8 warning
from cupy.sparse.linalg.solve import lsqr    # NOQA
from cupy.sparse.linalg.solve import lschol  # NOQA
