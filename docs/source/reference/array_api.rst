Python Array API Support
========================

The `Python array API standard <https://data-apis.org/array-api/latest/>`_ aims to provide a coherent set of
APIs for array and tensor libraries developed by the community to build upon. This solves the API fragmentation
issue across the community by offering concrete function signatures, semantics and scopes of coverage, enabling
writing backend-agnostic codes for better portability.

CuPy provides **experimental** support based on NumPy's `NEP-47 <https://numpy.org/neps/nep-0047-array-api-standard.html>`_,
which is in turn based on the draft standard to be finalized in 2021. All of the functionalities can be accessed
through the :mod:`cupy.array_api` namespace.

The key difference between NumPy and CuPy is that we are a GPU-only library, therefore CuPy users should be aware
of potential `device management <https://data-apis.org/array-api/latest/design_topics/device_support.html>`_ issues.
Same as in regular CuPy codes, the GPU-to-use can be specified via the :class:`~cupy.cuda.Device` objects, see
:ref:`device_management`.

.. toctree::
   :maxdepth: 2

   array_api_functions
   array_api_array
