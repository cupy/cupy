.. module:: cupyx.scipy.sparse

Sparse arrays (:mod:`cupyx.scipy.sparse`)
=========================================

.. Hint:: `SciPy API Reference: Sparse arrays (scipy.sparse) <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_

CuPy sparse array package for numeric data, matching
`SciPy's sparse package <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_.
It builds on `cuSPARSE <https://developer.nvidia.com/cusparse>`_ for
high-performance sparse linear algebra on the GPU.  Refer to the SciPy
documentation for usage; the sections below cover the CuPy-specific
differences.

.. note::

   Prefer the sparse **array** classes (``*_array``) for new code.  They
   are new in CuPy and follow NumPy-like semantics (``*`` is element-wise;
   use ``@`` for matrix multiplication).  SciPy plans to deprecate sparse
   **matrices** (``*_matrix``) in favor of sparse arrays and CuPy will
   follow.  When porting existing code, see SciPy's
   `Migration from spmatrix to sparray <https://docs.scipy.org/doc/scipy/reference/sparse.migration_to_sparray.html>`_
   guide.

CuPy differences from SciPy
---------------------------

* Formats: COO, CSR, CSC, and DIA only (no BSR, DOK, or LIL).
* Dimensions: 2-D for all formats, plus 1-D for
  :class:`~cupyx.scipy.sparse.coo_array` and
  :class:`~cupyx.scipy.sparse.csr_array`.  There is no n-D support, and
  hence no ``expand_dims``.
* Data dtypes: ``bool``, ``float32``, ``float64``, ``complex64``, and
  ``complex128``, matching what cuSPARSE supports.  SciPy additionally
  supports the integer dtypes.
* ``save_npz`` / ``load_npz`` are not implemented.

Index dtype (int32 / int64)
---------------------------

Like SciPy, CuPy sparse objects automatically choose the index
dtype (``indices``, ``indptr``, ``row``, ``col``) based on the
dimensions and index values:

* **int32** when all index values and dimensions fit in a 32-bit
  integer (the common case).
* **int64** when any dimension or index value exceeds
  ``2**31 - 1``.

The dtype is chosen by :func:`~cupyx.scipy.sparse.get_index_dtype`
(mirroring SciPy's logic) and is preserved through format conversions,
arithmetic, and indexing.  As in SciPy, sparse **array** constructors
keep the dtype of index arrays you pass in, while sparse **matrix**
constructors may downcast int64 indices to int32 when the values fit.

Operations that delegate to cuSPARSE use the native Generic API
(``SpMatDescr``) for int64 where available, with pure-CuPy
fallbacks for legacy int32-only APIs (e.g., ``csr2cscEx2``,
``xcoo2csr``, ``csrgeam2``).

Known limitations
~~~~~~~~~~~~~~~~~

The following operations are int32-only and raise ``ValueError``
when called on a sparse object with int64 indices:

* :func:`cupyx.scipy.sparse.linalg.spsolve` -- the underlying
  ``cusolverSp<t>csrlsvqr`` routine has no int64 overload.
* :func:`cupyx.scipy.sparse.linalg.spilu` -- the underlying
  ``cusparse<t>csrilu02`` routine has no int64 overload.
* :func:`cupyx.scipy.sparse.linalg.spsolve_triangular` on CUDA
  builds older than 12.0, where the dispatch falls back to
  ``cusparse<t>csrsm2``.  On CUDA 12.0+ it uses ``cusparseSpSM``
  (Generic API), which supports int64.

Conversion to/from SciPy
------------------------

CuPy and SciPy sparse objects are not implicitly convertible.
SciPy functions cannot take ``cupyx.scipy.sparse`` objects as inputs,
and vice versa.

- To convert SciPy sparse arrays/matrices to CuPy, pass them to the
  matching CuPy constructor such as :class:`~cupyx.scipy.sparse.csr_array`
  or :class:`~cupyx.scipy.sparse.csr_matrix`.
- To convert CuPy sparse objects to SciPy, use their
  :meth:`~cupyx.scipy.sparse.csr_array.get` method.  Array instances
  return a SciPy ``*_array``; matrix instances return a SciPy
  ``*_matrix``.

Converting between CuPy and SciPy incurs host-device data transfer,
which is costly.

Conversion to/from CuPy ndarrays
--------------------------------

- To convert a CuPy ndarray to a sparse object, pass it to a sparse
  constructor such as :class:`~cupyx.scipy.sparse.csr_array`.
- To convert a sparse object to a dense CuPy ndarray, use
  :meth:`~cupyx.scipy.sparse.csr_array.toarray`.

Converting between CuPy ndarray and CuPy sparse objects does not
incur host-device transfer; the data stays on the GPU.

Contents
--------

Sparse array classes
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   coo_array
   csc_array
   csr_array
   dia_array
   sparray


Sparse matrix classes
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   coo_matrix
   csc_matrix
   csr_matrix
   dia_matrix
   spmatrix


Building sparse arrays
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   eye_array
   diags_array
   block_array
   random_array


Building sparse matrices
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   eye
   identity
   diags
   spdiags
   bmat
   rand
   random


Combining and manipulating
~~~~~~~~~~~~~~~~~~~~~~~~~~

As in SciPy, these preserve the input type: the result is a sparse array
if any input is a sparse array, and a sparse matrix otherwise.  (The
``*_array`` builders listed above always return sparse arrays.)
``matrix_transpose``, ``permute_dims``, and ``swapaxes`` are 2-D only in
CuPy.

.. autosummary::
   :toctree: generated/

   kron
   kronsum
   hstack
   vstack
   block_diag
   tril
   triu
   matrix_transpose
   permute_dims
   swapaxes


Sparse tools
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   find
   get_index_dtype
   safely_cast_index_arrays


Identifying sparse arrays and matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As in SciPy, ``issparse`` accepts both sparse arrays and sparse matrices,
while ``isspmatrix`` and the per-format ``isspmatrix_*`` checks are true
only for sparse matrices.  Use ``isinstance(x, csr_array)`` and friends to
test for a sparse array of a given format.

.. autosummary::
   :toctree: generated/

   issparse
   isspmatrix
   isspmatrix_csc
   isspmatrix_csr
   isspmatrix_coo
   isspmatrix_dia


Submodules
~~~~~~~~~~

.. autosummary::

   csgraph - Compressed sparse graph routines
   linalg - Sparse linear algebra routines

Exceptions
~~~~~~~~~~

* :class:`scipy.sparse.SparseEfficiencyWarning`
* :class:`scipy.sparse.SparseWarning`
