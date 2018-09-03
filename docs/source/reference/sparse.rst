---------------
Sparse matrices
---------------

CuPy supports sparse matrices using `cuSPARSE <https://developer.nvidia.com/cusparse>`_.
These matrices have the same interfaces of `SciPy's sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_.

.. module:: cupyx.scipy.sparse

Conversion to/from SciPy sparse matrices
----------------------------------------

``cupyx.scipy.sparse.*_matrix`` and ``cupyx.scipy.sparse.*_matrix`` are not implicitly convertible to each other.
That means, SciPy functions cannot take ``cupyx.scipy.sparse.*_matrix`` objects as inputs, and vice versa.

- To convert SciPy sparse matrices to CuPy, pass it to the constructor of each CuPy sparse matrix class.
- To convert CuPy sparse matrices to SciPy, use :func:`get <cupyx.scipy.sparse.spmatrix.get>` method of each CuPy sparse matrix class.

Note that converting between CuPy and SciPy incurs data transfer between
the host (CPU) device and the GPU device, which is costly in terms of performance.

Sparse matrix classes
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.sparse.coo_matrix
   cupyx.scipy.sparse.csr_matrix
   cupyx.scipy.sparse.csc_matrix
   cupyx.scipy.sparse.dia_matrix
   cupyx.scipy.sparse.spmatrix


Functions
---------

Building sparse matrices
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.sparse.eye
   cupyx.scipy.sparse.identity
   cupyx.scipy.sparse.spdiags
   cupyx.scipy.sparse.rand
   cupyx.scipy.sparse.random


Identifying sparse matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.sparse.issparse
   cupyx.scipy.sparse.isspmatrix
   cupyx.scipy.sparse.isspmatrix_csc
   cupyx.scipy.sparse.isspmatrix_csr
   cupyx.scipy.sparse.isspmatrix_coo
   cupyx.scipy.sparse.isspmatrix_dia


Linear Algebra
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.sparse.linalg.lsqr
