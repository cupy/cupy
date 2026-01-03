Array API Compliant Object
==========================

.. note::
   ``cupy.array_api`` will be removed in CuPy v14 because its NumPy counterpart ``numpy.array_api`` has been removed.
   The root module ``cupy.*`` is now compatible with the Array API specification as it mirrors the NumPy v2 API.
   Use the `Array API compatibility library <https://data-apis.org/array-api-compat/>`_ to develop applications compatible with various array libraries, including CuPy, NumPy, and PyTorch.

:class:`~cupy.array_api._array_object.Array` is a wrapper class built upon :class:`cupy.ndarray`
to enforce strict compliance with the array API standard. See the
`documentation <https://data-apis.org/array-api/latest/API_specification/array_object.html>`_
for detail.

This object should not be constructed directly. Rather, use one of the
`creation functions <https://data-apis.org/array-api/latest/API_specification/creation_functions.html>`_,
such as :func:`cupy.array_api.asarray`.

.. currentmodule:: cupy.array_api._array_object

.. autosummary::
   :toctree: generated/

   Array
