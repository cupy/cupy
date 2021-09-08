Array API Complaint Object
==========================

:class:`~cupyx.array_api._array_object.Array` is a wrapper class built upon :class:`cupy.ndarray`
to enforce strict complaince with the array API standard. See the
`documentation <https://data-apis.org/array-api/latest/API_specification/array_object.html>`_
for detail.

This object should not be constructed directly. Rather, use one of the
`creation functions <https://data-apis.org/array-api/latest/API_specification/creation_functions.html>`_,
such as :func:`cupyx.array_api.asarray`.

.. currentmodule:: cupyx.array_api._array_object

.. autosummary::
   :toctree: generated/

   Array
