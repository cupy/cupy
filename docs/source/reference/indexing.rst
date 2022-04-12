Indexing routines
=================

.. Hint:: `NumPy API Reference: Indexing routines <https://numpy.org/doc/stable/reference/routines.indexing.html>`_

.. currentmodule:: cupy

Generating index arrays
-----------------------

.. autosummary::
   :toctree: generated/

   c_
   r_
   nonzero
   where
   indices
   mask_indices
   tril_indices
   tril_indices_from
   triu_indices
   triu_indices_from
   ix_
   ravel_multi_index
   unravel_index
   diag_indices
   diag_indices_from


Indexing-like operations
------------------------

.. autosummary::
   :toctree: generated/

   take
   take_along_axis
   choose
   compress
   diag
   diagonal
   select
   lib.stride_tricks.as_strided


Inserting data into arrays
--------------------------

.. autosummary::
   :toctree: generated/

   place
   put
   putmask
   fill_diagonal


Iterating over arrays
---------------------

.. autosummary::
   :toctree: generated/

   flatiter
