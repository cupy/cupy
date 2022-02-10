# flake8: NOQA
from cupyx.scipy.linalg._special_matrices import (
    tri, tril, triu, toeplitz, circulant, hankel,
    hadamard, leslie, kron, block_diag, companion,
    helmert, hilbert, dft,
    fiedler, fiedler_companion, convolution_matrix
)
from cupyx.scipy.linalg._solve_triangular import solve_triangular  # NOQA
from cupyx.scipy.linalg._decomp_lu import lu, lu_factor, lu_solve  # NOQA
