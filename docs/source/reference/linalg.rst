Linear Algebra
==============

.. https://docs.scipy.org/doc/numpy/reference/routines.linalg.html

Matrix and vector products
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   
   cupy.cross
   cupy.dot
   cupy.vdot
   cupy.inner
   cupy.outer
   cupy.matmul
   cupy.tensordot
   cupy.einsum
   cupy.linalg.matrix_power
   cupy.kron
   
   cupyx.scipy.linalg.kron

Decompositions
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.linalg.cholesky
   cupy.linalg.qr
   cupy.linalg.svd

Matrix eigenvalues
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.linalg.eigh
   cupy.linalg.eigvalsh

Norms etc.
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.linalg.det
   cupy.linalg.norm
   cupy.linalg.matrix_rank
   cupy.linalg.slogdet
   cupy.trace


Solving linear equations
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.linalg.solve
   cupy.linalg.tensorsolve
   cupy.linalg.lstsq
   cupy.linalg.inv
   cupy.linalg.pinv
   cupy.linalg.tensorinv

   cupyx.scipy.linalg.lu_factor
   cupyx.scipy.linalg.lu_solve
   cupyx.scipy.linalg.solve_triangular

Special Matrices
----------------

.. autosummary::

   :toctree: generated/
   :nosignatures:

   cupy.tri
   cupy.tril
   cupy.triu

   cupyx.scipy.linalg.tri
   cupyx.scipy.linalg.tril
   cupyx.scipy.linalg.triu
   cupyx.scipy.linalg.toeplitz
   cupyx.scipy.linalg.circulant
   cupyx.scipy.linalg.hankel
   cupyx.scipy.linalg.hadamard
   cupyx.scipy.linalg.leslie
   cupyx.scipy.linalg.block_diag
   cupyx.scipy.linalg.companion
   cupyx.scipy.linalg.helmert
   cupyx.scipy.linalg.hilbert
   cupyx.scipy.linalg.dft
   cupyx.scipy.linalg.fiedler
   cupyx.scipy.linalg.fiedler_companion
   cupyx.scipy.linalg.convolution_matrix
