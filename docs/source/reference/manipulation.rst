Array Manipulation Routines
===========================

.. https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html

Basic operations
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.copyto


Changing array shape
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.reshape
   cupy.ravel


Transpose-like operations
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.moveaxis
   cupy.rollaxis
   cupy.swapaxes
   cupy.transpose

.. seealso::
   :attr:`cupy.ndarray.T`

Changing number of dimensions
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.atleast_1d
   cupy.atleast_2d
   cupy.atleast_3d
   cupy.broadcast
   cupy.broadcast_to
   cupy.broadcast_arrays
   cupy.expand_dims
   cupy.squeeze


Changing kind of array
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.asarray
   cupy.asanyarray
   cupy.asfortranarray
   cupy.ascontiguousarray
   cupy.require


Joining arrays
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.concatenate
   cupy.stack
   cupy.column_stack
   cupy.dstack
   cupy.hstack
   cupy.vstack


Splitting arrays
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.split
   cupy.array_split
   cupy.dsplit
   cupy.hsplit
   cupy.vsplit


Tiling arrays
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.tile
   cupy.repeat


Adding and removing elements
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.unique


Rearranging elements
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.flip
   cupy.fliplr
   cupy.flipud
   cupy.reshape
   cupy.roll
   cupy.rot90
