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

Sparse matrix classes
---------------------

.. autosummary::
   :toctree: generated/

   coo_matrix
   csc_matrix
   csr_matrix
   dia_matrix
   spmatrix


Functions
---------

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


.. module:: cupyx.scipy.sparse.linalg

Linear Algebra (:mod:`cupyx.scipy.sparse.linalg`)
-------------------------------------------------

.. Hint:: `SciPy API Reference: Sparse linear algebra (scipy.sparse.linalg) <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_

Abstract linear operators
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   LinearOperator
   aslinearoperator


Matrix norms
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   cupyx.scipy.sparse.linalg.norm


Solving linear problems
~~~~~~~~~~~~~~~~~~~~~~~

Direct methods for linear equation systems:

.. autosummary::
   :toctree: generated/

   spsolve
   spsolve_triangular
   factorized

Iterative methods for linear equation systems:

.. autosummary::
   :toctree: generated/

   cg
   gmres

Iterative methods for least-squares problems:

.. autosummary::
   :toctree: generated/

   lsqr


Matrix factorizations
~~~~~~~~~~~~~~~~~~~~~

Eigenvalue problems:

.. autosummary::
   :toctree: generated/

   eigsh
   lobpcg

Singular values problems:

.. autosummary::
   :toctree: generated/

   svds

Complete or incomplete LU factorizations:

.. autosummary::
   :toctree: generated/

   splu
   spilu
