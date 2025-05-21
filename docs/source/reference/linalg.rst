.. module:: cupy.linalg

Linear algebra (:mod:`cupy.linalg`)
===================================

.. Hint:: `NumPy API Reference: Linear algebra (numpy.linalg) <https://numpy.org/doc/stable/reference/routines.linalg.html>`_

.. seealso:: :doc:`scipy_linalg`

The ``@`` operator
------------------

The ``@`` operator is preferable to other methods when computing the matrix product between 2d arrays.
The :func:`cupy.matmul` function implements the ``@`` operator.


.. currentmodule:: cupy

Matrix and vector products
--------------------------

.. autosummary::
   :toctree: generated/

   dot
   # linalg.multi_dot
   vdot
   # vecdot
   # linalg.vecdot
   inner
   outer
   # linalg.outer
   matmul
   # linalg.matmul (Array API compatible location)
   # matvec
   # vecmat
   tensordot
   # linalg.tensordot (Array API compatible location)
   einsum
   # einsum_path
   linalg.matrix_power
   kron
   linalg.cross

Decompositions
--------------

.. autosummary::
   :toctree: generated/

   linalg.cholesky
   linalg.qr
   linalg.svd
   # linalg.svdvals

Matrix eigenvalues
------------------

.. autosummary::
   :toctree: generated/

   # linalg.eig
   linalg.eigh
   # linalg.eigvals
   linalg.eigvalsh

Norms and other numbers
-----------------------

.. autosummary::
   :toctree: generated/

   linalg.norm
   # linalg.matrix_norm (Array API compatible)
   # linalg.vector_norm (Array API compatible)
   # linalg.cond
   linalg.det
   linalg.matrix_rank
   linalg.slogdet
   trace
   # linalg.trace (Array API compatible)

Solving equations and inverting matrices
----------------------------------------

.. autosummary::
   :toctree: generated/

   linalg.solve
   linalg.tensorsolve
   linalg.lstsq
   linalg.inv
   linalg.pinv
   linalg.tensorinv

Other matrix operations
-----------------------
.. autosummary::
   :toctree: generated/

   diagonal
   # linalg.diagonal (Array API compatible)
   # linalg.matrix_transpose (Array API compatible)
