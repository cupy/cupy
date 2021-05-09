Array manipulation routines
===========================

.. Hint:: `NumPy API Reference: Array manipulation routines <https://numpy.org/doc/stable/reference/routines.array-manipulation.html>`_

.. currentmodule:: cupy

Basic operations
----------------

.. autosummary::
   :toctree: generated/

   copyto
   shape


Changing array shape
--------------------

.. autosummary::
   :toctree: generated/

   reshape
   ravel

.. seealso:: :attr:`cupy.ndarray.flat` and :func:`cupy.ndarray.flatten`

Transpose-like operations
-------------------------

.. autosummary::
   :toctree: generated/

   moveaxis
   rollaxis
   swapaxes
   transpose

.. seealso:: :attr:`cupy.ndarray.T`

Changing number of dimensions
-----------------------------

.. autosummary::
   :toctree: generated/

   atleast_1d
   atleast_2d
   atleast_3d
   broadcast
   broadcast_to
   broadcast_arrays
   expand_dims
   squeeze


Changing kind of array
----------------------

.. autosummary::
   :toctree: generated/

   asarray
   asanyarray
   asfortranarray
   ascontiguousarray
   require


Joining arrays
--------------

.. autosummary::
   :toctree: generated/

   concatenate
   stack
   vstack
   hstack
   dstack
   column_stack


Splitting arrays
----------------

.. autosummary::
   :toctree: generated/

   split
   array_split
   dsplit
   hsplit
   vsplit


Tiling arrays
-------------

.. autosummary::
   :toctree: generated/

   tile
   repeat


Adding and removing elements
----------------------------

.. autosummary::
   :toctree: generated/

   append
   resize
   unique
   trim_zeros


Rearranging elements
--------------------

.. autosummary::
   :toctree: generated/

   flip
   fliplr
   flipud
   reshape
   roll
   rot90
