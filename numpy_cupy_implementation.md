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
- **Array API Compliance:** New functions should use positional-only arguments (`/`) and keyword-only arguments where specified by the standard.

### 3. Testing
- Use `cupy.testing.numpy_cupy_allclose` to compare against NumPy.
- Ensure tests cover various dtypes (including complex), shapes, and edge cases (e.g., empty arrays, different axes).
- For new functions in `linalg`, ensure they are tested within the `cupy.linalg` namespace.

## Specific Implementations

### `vecdot`
- **Goal:** Implement `numpy.vecdot` (introduced in NumPy 2.0).
- **Signature:** `vecdot(x1, x2, /, *, axis=-1, out=None, **kwargs)`
- **Implementation:** Used `cupy._core._gufuncs._GUFunc` with signature `(n),(n)->()`. The core function conjugates `x1` if complex and uses `cupy.multiply(x1, x2).sum(axis=-1)`.

### `multi_dot`
- **Goal:** Optimized dot product of 2+ arrays.
- **Implementation:** Implemented the Matrix Chain Multiplication order algorithm (dynamic programming) to find the optimal parenthesization.
- **Optimization:** For `n=3`, a simplified cost comparison is used for speed. For `n>3`, the standard DP approach is applied.
- **Handling 1D:** Explicitly handles 1D vectors at the start/end of the chain by treating them as row/column matrices and then flattening the result.

### `svdvals`, `matrix_norm`, `vector_norm`
- **Goal:** Array API compliant wrappers.
- **Implementation:** `svdvals` calls `svd(..., compute_uv=False)`. `matrix_norm` and `vector_norm` provide cleaner interfaces to the general `norm` function with Array API specific defaults.

### `linalg.trace` & `linalg.diagonal`
- **Goal:** Namespace-specific versions with different defaults.
- **Implementation:** Unlike top-level `cupy.trace`, `cupy.linalg.trace` operates on the last two axes (`-2, -1`) by default to comply with the Array API.

## Mistakes Made & Lessons Learned

### Technical Mistakes
- **`functools.reduce` with `out`:** Initially attempted `functools.reduce(cupy.dot, arrays, out=out)`. `reduce` does not support the `out` parameter.
  - *Lesson:* Be cautious when mixing standard Python functional tools with specialized library parameters. Use `cupy.copyto` if manual result placement is needed.
- **Missing Positional-Only Markers:** Initially forgot the `/` marker in `svdvals(x, /)`.
  - *Lesson:* Always double-check the Array API specification for strict signature requirements.
- **Code Duplication:** Duplicated fallback logic for specialized kernels in multiple places within `norm`.
  - *Lesson:* Refactor common recovery logic into private helpers (e.g., `_norm_fallback`) to improve maintainability.

### Environmental Robustness
- **Specialized Kernel Failures:** Encountered `CompileException` (cannot open source file "cstddef") on Windows with CUDA 12.8 when running specialized reduction kernels in `norm`.
  - *Lesson:* Always implement a "slow but sure" fallback to basic operations (like `sum` of squares) for specialized kernels. This ensures the library remains functional even in broken compilation environments.

### Process Mistakes
- **PR Target:** Initially created the PR against the fork's `main` branch instead of the upstream `cupy/cupy`.
  - *Lesson:* When working with forks, explicitly use `--repo cupy/cupy` with the `gh` CLI and ensure the `--base` and `--head` (with `user:branch` syntax) are correct.
- **Branch Management:** Managing multiple feature sets on a single branch led to merge conflicts when trying to isolate a specific set of changes for a PR.
  - *Lesson:* Use isolated feature branches (`linalg-compat`, `creation-compat`) from the start, and only merge into a combined development branch once isolated PRs are ready.
- **Documentation Visibility:** New functions were implemented but remained commented out in `linalg.rst`.
  - *Lesson:* Implementation is not complete until the `.rst` documentation files are updated to expose the new APIs to users.
