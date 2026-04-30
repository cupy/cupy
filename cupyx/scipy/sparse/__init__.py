from __future__ import annotations

# Base classes / type predicates
from cupyx.scipy.sparse._base import _spbase  # NOQA
from cupyx.scipy.sparse._base import issparse  # NOQA
from cupyx.scipy.sparse._base import isspmatrix  # NOQA
from cupyx.scipy.sparse._base import sparray  # NOQA
from cupyx.scipy.sparse._base import spmatrix  # NOQA
from cupyx.scipy.sparse._base import SparseWarning  # NOQA
from cupyx.scipy.sparse._base import SparseEfficiencyWarning  # NOQA

# Format classes (array + matrix variants)
from cupyx.scipy.sparse._coo import coo_array  # NOQA
from cupyx.scipy.sparse._coo import coo_matrix  # NOQA
from cupyx.scipy.sparse._coo import isspmatrix_coo  # NOQA
from cupyx.scipy.sparse._csc import csc_array  # NOQA
from cupyx.scipy.sparse._csc import csc_matrix  # NOQA
from cupyx.scipy.sparse._csc import isspmatrix_csc  # NOQA
from cupyx.scipy.sparse._csr import csr_array  # NOQA
from cupyx.scipy.sparse._csr import csr_matrix  # NOQA
from cupyx.scipy.sparse._csr import isspmatrix_csr  # NOQA
from cupyx.scipy.sparse._dia import dia_array  # NOQA
from cupyx.scipy.sparse._dia import dia_matrix  # NOQA
from cupyx.scipy.sparse._dia import isspmatrix_dia  # NOQA

# Construction (matrix-style: kept for back-compat)
from cupyx.scipy.sparse._construct import eye  # NOQA
from cupyx.scipy.sparse._construct import identity  # NOQA
from cupyx.scipy.sparse._construct import rand  # NOQA
from cupyx.scipy.sparse._construct import random  # NOQA
from cupyx.scipy.sparse._construct import spdiags  # NOQA
from cupyx.scipy.sparse._construct import diags  # NOQA
from cupyx.scipy.sparse._construct import bmat  # NOQA

# Construction (array-style; preferred going forward)
from cupyx.scipy.sparse._construct import block_array  # NOQA
from cupyx.scipy.sparse._construct import block_diag  # NOQA
from cupyx.scipy.sparse._construct import diags_array  # NOQA
from cupyx.scipy.sparse._construct import eye_array  # NOQA
from cupyx.scipy.sparse._construct import random_array  # NOQA

# Combining (type-aware: returns array when all inputs are arrays)
from cupyx.scipy.sparse._construct import hstack  # NOQA
from cupyx.scipy.sparse._construct import vstack  # NOQA
from cupyx.scipy.sparse._construct import kron  # NOQA
from cupyx.scipy.sparse._construct import kronsum  # NOQA

# Axis manipulation (2-D only in CuPy; SciPy supports nD).
# ``expand_dims`` is omitted because it would require nD sparse arrays.
from cupyx.scipy.sparse._construct import matrix_transpose  # NOQA
from cupyx.scipy.sparse._construct import permute_dims  # NOQA
from cupyx.scipy.sparse._construct import swapaxes  # NOQA

# Extraction
from cupyx.scipy.sparse._extract import find  # NOQA
from cupyx.scipy.sparse._extract import tril  # NOQA
from cupyx.scipy.sparse._extract import triu  # NOQA

# Index-dtype utilities (re-exported to match scipy.sparse)
from cupyx.scipy.sparse._sputils import get_index_dtype  # NOQA
from cupyx.scipy.sparse._sputils import safely_cast_index_arrays  # NOQA

# Not-yet-implemented in CuPy:
# - bsr_array / bsr_matrix (Block Sparse Row)
# - dok_array / dok_matrix (Dictionary of Keys)
# - lil_array / lil_matrix (List of Lists)
# - isspmatrix_bsr / isspmatrix_lil / isspmatrix_dok
# - save_npz / load_npz
# - expand_dims (would require nD sparse-array support)
