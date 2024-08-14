from cupyx.scipy.linalg._array_utils import bandwidth
from cupyx.scipy.linalg._decomp_lu import lu, lu_factor, lu_solve
from cupyx.scipy.linalg._matfuncs import expm, khatri_rao
from cupyx.scipy.linalg._solve_triangular import solve_triangular
from cupyx.scipy.linalg._special_matrices import (
    block_diag,
    circulant,
    companion,
    convolution_matrix,
    dft,
    fiedler,
    fiedler_companion,
    hadamard,
    hankel,
    helmert,
    hilbert,
    kron,
    leslie,
    toeplitz,
    tri,
    tril,
    triu,
)

# uarray backend support (NEP 31)
# The uarray feature for scipy.linalg is experimental.
# The interface can change in the future.
from cupyx.scipy.linalg._uarray import (
    __ua_convert__,
    __ua_domain__,
    __ua_function__,
)
