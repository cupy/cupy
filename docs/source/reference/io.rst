Input and output
================

.. Hint:: `NumPy API Reference: Input and output <https://numpy.org/doc/stable/reference/routines.io.html>`_

.. currentmodule:: cupy


NumPy binary files (npy, npz)
-----------------------------

.. autosummary::
   :toctree: generated/

   _io.npz.NpzFile
   _io.npz.BagObj
   load
   save
   savez
   savez_compressed
   # lib.npyio.NpzFile

Text files
-----------------------------

.. autosummary::
   :toctree: generated/

   loadtxt
   savetxt
   genfromtxt
   # fromregex
   fromstring
   # ndarray.tofile
   # ndarray.tolist

.. seealso:: :attr:`cupy.ndarray.tofile` and :func:`cupy.ndarray.tolist`

Raw binary files
----------------

.. autosummary::

   fromfile
   # ndarray.tofile

.. seealso:: :attr:`cupy.ndarray.tofile`

String formatting
-----------------

.. autosummary::
   :toctree: generated/

   array2string
   array_repr
   array_str
   format_float_positional
   format_float_scientific

Base-n representations
----------------------

.. autosummary::
   :toctree: generated/

   binary_repr
   base_repr
