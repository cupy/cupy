Array creation routines
=======================

.. Hint:: `NumPy API Reference: Array creation routines <https://numpy.org/doc/stable/reference/routines.array-creation.html>`_

.. currentmodule:: cupy

From shape or value
-------------------
.. autosummary::
   :toctree: generated/

   empty
   empty_like
   eye
   identity
   ones
   ones_like
   zeros
   zeros_like
   full
   full_like

From existing data
------------------
.. autosummary::
   :toctree: generated/

   array
   asarray
   asanyarray
   ascontiguousarray
   # asmatrix
   astype
   copy
   frombuffer
   from_dlpack
   fromfile
   fromfunction
   fromiter
   fromstring
   loadtxt


Numerical ranges
----------------
.. autosummary::
   :toctree: generated/

   arange
   linspace
   logspace
   # geomspace
   meshgrid
   mgrid
   ogrid

Building matrices
-----------------
.. autosummary::
   :toctree: generated/

   diag
   diagflat
   tri
   tril
   triu
   vander
