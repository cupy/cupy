# flake8: NOQA
from cupyx.scipy.linalg._special_matrices import (
    tri, tril, triu, toeplitz, circulant, hankel,
    hadamard, leslie, kron, block_diag, companion,
    helmert, hilbert, dft,
    fiedler, fiedler_companion, convolution_matrix
)
from cupyx.scipy.linalg._solve_triangular import solve_triangular  # NOQA
from cupyx.scipy.linalg._decomp_lu import lu, lu_factor, lu_solve  # NOQA

# uarray backend support (NEP 31)
# The uarray feature for scipy.linalg is experimental.
# The interface can change in the future.
from cupyx.scipy.linalg._uarray import __ua_convert__  # NOQA
from cupyx.scipy.linalg._uarray import __ua_domain__  # NOQA
from cupyx.scipy.linalg._uarray import __ua_function__  # NOQA
