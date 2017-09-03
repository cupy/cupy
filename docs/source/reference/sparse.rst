-------------
Sparse matrix
-------------

CuPy supports sparse matrices using `cuSPARSE <https://developer.nvidia.com/cusparse>`_.
These matrices have the same interfaces of `SciPy's sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_.

.. module:: cupy.sparse


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
