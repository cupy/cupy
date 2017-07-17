-------------
Sparse matrix
-------------

CuPy supports sparse matrix using `cuSPARSE <https://developer.nvidia.com/cusparse>`_.
These matrices have the same interfaces of `SciPy's sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.sparse.coo_matrix
   cupy.sparse.csr_matrix
   cupy.sparse.csc_matrix
   cupy.sparse.spmatrix
