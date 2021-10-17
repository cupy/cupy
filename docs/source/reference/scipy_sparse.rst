.. module:: cupyx.scipy.sparse

Sparse matrices (:mod:`cupyx.scipy.sparse`)
===========================================

.. Hint:: `SciPy API Reference: Sparse matrices (scipy.sparse) <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_

CuPy supports sparse matrices using `cuSPARSE <https://developer.nvidia.com/cusparse>`_.
These matrices have the same interfaces of `SciPy's sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_.

Conversion to/from SciPy sparse matrices
----------------------------------------

``cupyx.scipy.sparse.*_matrix`` and ``scipy.sparse.*_matrix`` are not implicitly convertible to each other.
That means, SciPy functions cannot take ``cupyx.scipy.sparse.*_matrix`` objects as inputs, and vice versa.

- To convert SciPy sparse matrices to CuPy, pass it to the constructor of each CuPy sparse matrix class.
- To convert CuPy sparse matrices to SciPy, use :func:`get <cupyx.scipy.sparse.spmatrix.get>` method of each CuPy sparse matrix class.

Note that converting between CuPy and SciPy incurs data transfer between
the host (CPU) device and the GPU device, which is costly in terms of performance.

Conversion to/from CuPy ndarrays
--------------------------------

- To convert CuPy ndarray to CuPy sparse matrices, pass it to the constructor of each CuPy sparse matrix class.
- To convert CuPy sparse matrices to CuPy ndarray, use ``toarray`` of each CuPy sparse matrix instance (e.g., :func:`cupyx.scipy.sparse.csr_matrix.toarray`).

Converting between CuPy ndarray and CuPy sparse matrices does not incur data transfer; it is copied inside the GPU device.

Contents
--------

Sparse matrix classes
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   coo_matrix
   csc_matrix
   csr_matrix
   dia_matrix
   spmatrix


Functions
~~~~~~~~~

Building sparse matrices:

.. autosummary::
   :toctree: generated/

   eye
   identity
   kron
   kronsum
   diags
   spdiags
   tril
   triu
   bmat
   hstack
   vstack
   rand
   random


Sparse matrix tools:

.. autosummary::
   :toctree: generated/

   find

Identifying sparse matrices:

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
