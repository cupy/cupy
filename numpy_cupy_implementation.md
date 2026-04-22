# CuPy NumPy Compatibility Implementation

This skill documents the patterns and lessons learned while implementing NumPy-compatible APIs in CuPy.

## General Patterns

### 1. Identifying Missing APIs
- Check the [CuPy Tracker Issue #6078](https://github.com/cupy/cupy/issues/6078) for the list of missing `numpy.*` APIs.
- Use `grep_search` to verify if an API is already implemented or has an alias.

### 2. Implementation Strategy
- **Aliases:** If the API is just a rename (e.g., `cumulative_sum` for `cumsum` in NumPy 2.0), implement it as an alias or a wrapper with a deprecation warning if applicable.
- **GUFuncs:** For functions that support broadcasting and specific axes (like `vecdot`), use `cupy._core._gufuncs._GUFunc`.
- **Complex Conjugation:** Be careful with functions like `vdot` and `vecdot` which conjugate the first argument for complex inputs.

### 3. Testing
- Use `cupy.testing.numpy_cupy_allclose` to compare against NumPy.
- Ensure tests cover various dtypes (including complex), shapes, and edge cases (e.g., empty arrays, different axes).

## Specific Implementations

### `vecdot`
- **Goal:** Implement `numpy.vecdot` (introduced in NumPy 2.0).
- **Signature:** `vecdot(x1, x2, /, *, axis=-1, out=None, **kwargs)`
- **Behavior:** Vector dot product over a specific axis. Conjugates the first argument if complex. Supports broadcasting on non-summation axes.
- **Implementation:** Used `cupy._core._gufuncs._GUFunc` with signature `(n),(n)->()`. The core function conjugates `x1` if complex and uses `cupy.multiply(x1, x2).sum(axis=-1)`.

### `cumulative_sum` & `cumulative_prod`
- **Goal:** Implement Array API compatible aliases for `cumsum` and `cumprod`.
- **Behavior:** These functions REQUIRE the `axis` argument if the input array has more than one dimension. This differs from `cumsum`/`cumprod` which flatten by default.
- **Implementation:** Wrappers around `cumsum`/`cumprod` that check `axis is None and x.ndim > 1`.

### `matrix_transpose`
- **Goal:** Export `matrix_transpose` to the top-level `cupy` namespace.
- **Implementation:** Added import to `cupy/__init__.py`. This function was already implemented in `cupy._manipulation.transpose`.

## Lessons Learned
- **Broadcasting in GUFuncs:** `_GUFunc` handles the transposition of the core axis to the end and broadcasting of outer dimensions automatically. The core function should always operate on the last axis.
- **Array API Strictness:** New NumPy 2.0 functions (often from Array API) sometimes have stricter requirements than their traditional counterparts (e.g., requiring `axis`).
- **Namespace Consistency:** Ensure new functions are exported in both the top-level `cupy` and relevant submodules (e.g., `cupy.linalg`) if applicable.
