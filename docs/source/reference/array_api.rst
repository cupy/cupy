Python Array API Support
========================

The `Python array API standard <https://data-apis.org/array-api/2021.12/>`_ aims to provide a coherent set of
APIs for array and tensor libraries developed by the community to build upon. This solves the API fragmentation
issue across the community by offering concrete function signatures, semantics and scopes of coverage, enabling
writing backend-agnostic codes for better portability.

Like NumPy, CuPy is compatible with the Array API. However, we recommand using the
`Array API compatibility library <https://data-apis.org/array-api-compat/>`_
library when developing applications or libraries using the Array API for interoperability.
Similar to NumPy, the CuPy main namespace may have to lag behind changes and the compatibility library is used
by several downstream projects for this purpose.
