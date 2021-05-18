.. module:: cupy.linalg

Linear algebra (:mod:`cupy.linalg`)
===================================

.. Hint:: `NumPy API Reference: Linear algebra (numpy.linalg) <https://numpy.org/doc/stable/reference/routines.linalg.html>`_

.. seealso:: :doc:`scipy_linalg`

.. currentmodule:: cupy

Matrix and vector products
--------------------------

.. autosummary::
   :toctree: generated/

   dot
   vdot
   inner
   outer
   matmul
   tensordot
   einsum
   linalg.matrix_power
   kron

Decompositions
--------------

.. autosummary::
   :toctree: generated/

   linalg.cholesky
   linalg.qr
   linalg.svd

Matrix eigenvalues
------------------

.. autosummary::
   :toctree: generated/

   linalg.eigh
   linalg.eigvalsh

Norms and other numbers
-----------------------

.. autosummary::
   :toctree: generated/

   linalg.norm
   linalg.det
   linalg.matrix_rank
   linalg.slogdet
   trace


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
