.. module:: cupyx.scipy.sparse.linalg

Sparse linear algebra (:mod:`cupyx.scipy.sparse.linalg`)
========================================================

.. Hint:: `SciPy API Reference: Sparse linear algebra (scipy.sparse.linalg) <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_

Abstract linear operators
-------------------------

.. autosummary::
   :toctree: generated/

   LinearOperator
   aslinearoperator


Matrix norms
------------

.. autosummary::
   :toctree: generated/

   norm


Solving linear problems
-----------------------

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
   cgs
   minres

Iterative methods for least-squares problems:

.. autosummary::
   :toctree: generated/

   lsqr
   lsmr


Matrix factorizations
---------------------

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
   SuperLU
