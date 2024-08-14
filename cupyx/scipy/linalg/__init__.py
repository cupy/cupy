from cupyx.scipy.linalg._array_utils import bandwidth  # NOQA
from cupyx.scipy.linalg._decomp_lu import lu, lu_factor, lu_solve  # NOQA
from cupyx.scipy.linalg._matfuncs import (
    expm,  # NOQA
    khatri_rao,  # NOQA
)
from cupyx.scipy.linalg._solve_triangular import solve_triangular  # NOQA
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
)  # NOQA

# uarray backend support (NEP 31)
# The uarray feature for scipy.linalg is experimental.
# The interface can change in the future.
from cupyx.scipy.linalg._uarray import (
    __ua_convert__,  # NOQA
    __ua_domain__,  # NOQA
    __ua_function__,  # NOQA
)
