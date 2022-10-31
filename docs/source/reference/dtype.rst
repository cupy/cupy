Data type routines
==================

.. Hint:: `NumPy API Reference: Data type routines <https://numpy.org/doc/stable/reference/routines.dtype.html>`_

.. currentmodule:: cupy

.. autosummary::
   :toctree: generated/

   can_cast
   min_scalar_type
   result_type
   common_type

.. csv-table::
   :align: left

   ``promote_types`` (alias of :func:`numpy.promote_types`)
   ``obj2sctype`` (alias of :func:`numpy.obj2sctype`)

Creating data types
-------------------

.. csv-table::
   :align: left

   ``dtype`` (alias of :class:`numpy.dtype`)
   ``format_parser`` (alias of :class:`numpy.format_parser`)

Data type information
---------------------

.. csv-table::
   :align: left

   ``finfo`` (alias of :class:`numpy.finfo`)
   ``iinfo`` (alias of :class:`numpy.iinfo`)
   ``MachAr`` (alias of :class:`numpy.MachAr`)

Data type testing
-----------------

.. csv-table::
   :align: left

   ``issctype`` (alias of :func:`numpy.issctype`)
   ``issubdtype`` (alias of :func:`numpy.issubdtype`)
   ``issubsctype`` (alias of :func:`numpy.issubsctype`)
   ``issubclass_`` (alias of :func:`numpy.issubclass_`)
   ``find_common_type`` (alias of :func:`numpy.find_common_type`)

Miscellaneous
-------------

.. csv-table::
   :align: left

   ``typename`` (alias of :func:`numpy.typename`)
   ``sctype2char`` (alias of :func:`numpy.sctype2char`)
   ``mintypecode`` (alias of :func:`numpy.mintypecode`)
