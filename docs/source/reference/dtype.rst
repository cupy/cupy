Data type routines
==================

.. Hint:: `NumPy API Reference: Data type routines <https://numpy.org/doc/stable/reference/routines.dtype.html>`_

.. currentmodule:: cupy

.. autosummary::
   :toctree: generated/

   can_cast
   min_scalar_type
   make_aligned_dtype
   result_type
   common_type

.. csv-table::
   :align: left

   ``promote_types`` (alias of :func:`numpy.promote_types`)

Creating data types
-------------------

.. csv-table::
   :align: left

   ``dtype`` (alias of :class:`numpy.dtype`)
   ``make_aligned_dtype`` CuPy utility to create structured dtypes with sufficient alignment for GPU use.
   ``format_parser`` (alias of :class:`numpy.rec.format_parser`)

Data type information
---------------------

.. csv-table::
   :align: left

   ``finfo`` (alias of :class:`numpy.finfo`)
   ``iinfo`` (alias of :class:`numpy.iinfo`)

Data type testing
-----------------

.. csv-table::
   :align: left

   ``issubdtype`` (alias of :func:`numpy.issubdtype`)
   ``isdtype`` (alias of :func:`numpy.isdtype`)

Miscellaneous
-------------

.. csv-table::
   :align: left

   ``typename`` (alias of :func:`numpy.typename`)
   ``mintypecode`` (alias of :func:`numpy.mintypecode`)
