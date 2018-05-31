---------------
Sparse matrices
---------------

CuPy supports sparse matrices using `cuSPARSE <https://developer.nvidia.com/cusparse>`_.
These matrices have the same interfaces of `SciPy's sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_.

.. module:: cupy.sparse

Conversion to/from SciPy sparse matrices
----------------------------------------

``cupy.sparse.*_matrix`` and ``scipy.sparse.*_matrix`` are not implicitly convertible to each other.
That means, SciPy functions cannot take ``cupy.sparse.*_matrix`` objects as inputs, and vice versa.

- To convert SciPy sparse matrices to CuPy, pass it to the constructor of each CuPy sparse matrix class.
- To convert CuPy sparse matrices to SciPy, use :func:`get <cupy.sparse.spmatrix.get>` method of each CuPy sparse matrix class.

Note that converting between CuPy and SciPy incurs data transfer between
the host (CPU) device and the GPU device, which is costly in terms of performance.

Sparse matrix classes
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.sparse.coo_matrix
   cupy.sparse.csr_matrix
   cupy.sparse.csc_matrix
   cupy.sparse.dia_matrix
   cupy.sparse.spmatrix


Functions
---------

Building sparse matrices
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.sparse.eye
   cupy.sparse.identity
   cupy.sparse.spdiags
   cupy.sparse.rand
   cupy.sparse.random


Identifying sparse matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.sparse.issparse
   cupy.sparse.isspmatrix
   cupy.sparse.isspmatrix_csc
   cupy.sparse.isspmatrix_csr
   cupy.sparse.isspmatrix_coo
   cupy.sparse.isspmatrix_dia


Linear Algebra
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.sparse.linalg.lsqr
