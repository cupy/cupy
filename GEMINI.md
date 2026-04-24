# CuPy Development Guide

CuPy is a NumPy/SciPy-compatible array library for GPU-accelerated computing with Python. It supports NVIDIA CUDA and
AMD ROCm as backends.

## Project Overview

- **Core Purpose:** Drop-in replacement for NumPy and SciPy on GPUs.
- **Main Technologies:** Python, Cython, C++, CUDA, ROCm (HIP).
- **Architecture:**
    - `cupy/`: High-level NumPy-compatible APIs.
    - `cupyx/`: SciPy-compatible APIs (`cupyx.scipy`) and experimental features.
    - `cupy_backends/`: Low-level abstraction layer for GPU runtimes (CUDA and HIP).
    - `cupy_builder/`: Custom build system logic located in `install/`.

## Building and Running

### Prerequisites

- CUDA Toolkit or ROCm environment.
- Python 3.10+
- Cython and NumPy (see `pyproject.toml` for versions).

### Build Commands

- **Install in development mode:**
  ```powershell
  pip install -e .
  ```
- **Build extensions in-place:**
  ```powershell
  python setup.py build_ext --inplace
  ```

### Running Tests

Testing requires `pytest`. Many tests require a GPU.

- **Run all tests:**
  ```powershell
  pytest tests/cupy_tests
  ```
- **Run specific test file:**
  ```powershell
  pytest tests/cupy_tests/core_tests/test_ndarray_indexing.py
  ```
- **Multi-GPU Testing:**
  Use `CUPY_TEST_GPU_LIMIT` to enable automatic GPU rotation with `pytest-xdist`:
  ```powershell
  $env:CUPY_TEST_GPU_LIMIT=2
  pytest -n 2 tests/cupy_tests
  ```

## Development Conventions

### Code Style and Linting

The project uses `ruff`, `mypy`, and `autopep8`.

- **Run all linters via pre-commit:**
  ```powershell
  pre-commit run --all-files
  ```
- **Manual Linting:**
    - `ruff check .`
    - `mypy .`
    - `cython-lint .` (for `.pyx` and `.pxd` files)

### Testing Practices

- Use `cupy.testing` module for GPU-specific assertions and decorators (e.g., `@cupy.testing.numpy_cupy_allclose`).
- Most tests are designed to compare results against NumPy to ensure compatibility.

## Key Directories

- `cupy/`: Main package containing ndarray implementation, ufuncs, and NumPy-compatible modules (linalg, fft, random,
  etc.).
- `cupyx/`: Extension package for SciPy-compatible modules (`scipy.linalg`, `scipy.signal`, `scipy.sparse`, etc.) and
  JIT compiler.
- `cupy_backends/`: Contains the Cython wrappers for CUDA and HIP APIs.
- `tests/`: Extensive test suite divided by subpackage.
- `install/`: Contains `cupy_builder`, the logic for detecting CUDA/ROCm and configuring the build.
- `.pfnci/`: Configuration and scripts for Continuous Integration.

## Usage Tips

- **GPU Selection:** Use `CUDA_VISIBLE_DEVICES` environment variable to restrict which GPUs CuPy can see.
- **Checking Availability:** `cupy.is_available()` returns `True` if a compatible GPU and driver are detected.

## CuPy Pull Request Review Insights & Conventions

This section summarizes project-specific patterns and "unwritten rules" distilled from past closed pull requests and
code reviews.

### 🚀 Performance & Synchronization

- **Avoid DtoH Synchronization:** Any logic that performs a GPU check (e.g., `cupy.all(indices[:-1] <= indices[1:])`)
  and branches on the result in Python is heavily scrutinized. Stay on the device as long as possible.
- **Layout Awareness:** Kernels and APIs must handle both C-contiguous and F-contiguous layouts correctly.
- **Hardware Atomicity:** Avoid marking 16-byte types (e.g., `complex128`) as "primitive" in `NumericTraits` to prevent
  torn reads/writes in advanced CUB/Thrust algorithms.
- **Hardware Limits:** Accounting for `gridDim` overflows and providing workarounds (like grid-stride loops) is
  mandatory for kernels.
- **Internal Sync Tools:** Use project-specific tools like `pymutex` (in `cupy.cuda.memory`) for thread-safe memory
  management.

### 🧹 Resource Management & Memory

- **Robust Destruction:** Use `weakref.finalize` for low-level resource management to avoid `ModuleNotFoundError` during
  interpreter shutdown.
- **Initialization Performance:** Prefer `cupy.empty` over `cupy.zeros` if the output array is guaranteed to be fully
  overwritten.
- **Reference Cycles:** In multi-threaded Cython code, explicitly break reference cycles in `__dealloc__` to prevent
  memory leaks in TLS.
- **Allocation Minimization:** Favor algorithms that reduce working memory from $O(kN)$ to $O(N)$ (e.g., combining axis
  shifts into a single `cupy.roll`).

### 🏗️ Module Architecture & API

- **Lazy Loading:** Submodules like `cupyx` should be imported lazily (inside functions) to keep `import cupy` fast.
- **SciPy Alignment:** Mirror SciPy's deprecation timelines. Implement replacement APIs first, then deprecate.
- **External Logic:** Use git submodules (e.g., `xsf`) for sharing C++ logic with SciPy instead of manual vendoring.

### 🛠️ Cython, JIT & C++ Implementation

- **Cython as Dispatcher:** `.pyx` files should primarily dispatch between optimized "fast paths" (CUB, cuBLAS) and
  generic fallback paths.
- **Fast-Path Dispatching:** Use `try_fastcall` patterns to bypass `GUFunc` overhead for critical, high-frequency
  operations on small matrices.
- **Computational Precision:** Avoid automatic promotion to `float64` in kernels. Use `float32` for `float32` inputs.

### 📊 Sparse Matrix & Distributed Logic

- **Format Conversion Bottlenecks:** Avoid CSR ↔ CSC conversions. Implement kernels that handle the provided format
  directly.
- **Two-Phase Exchange:** For sparse distributed collectives (e.g., `all_to_all`), exchange metadata (sizes) first, then
  data, to avoid NCCL hangs.

### 🧪 Testing & CI Best Practices

- **Dtype Coverage:** Always explicitly test complex types (`'D'`, `'F'`).
- **Scale Testing:** Verify indexing and memory logic with arrays up to $10^9$ elements.
- **Dtype-Dependent Tolerances:** Use specific mappings in tests (e.g., `rtol={float32: 1e-5, default: 1e-7}`) to allow
  for GPU variance while maintaining high-accuracy paths.

## NumPy Compatibility Development Cycle

When implementing or updating `numpy.*` compatible APIs (especially for NumPy 2.0 and Array API compatibility):

1. **Identify Targets:**
    - Consult the tracking issue (e.g., [#6078](https://github.com/cupy/cupy/issues/6078)) for missing or incompatible
      APIs.
    - Make sure no one else is working on the same issue (this issue's comments and the PR are a good place to check).
    - Add a comment to the issue to indicate that you are working on it.
2. **Research Specification:** Verify the exact behavior, signature, and requirements (e.g., broadcasting, conjugation,
   mandatory arguments) from the [NumPy documentation](https://numpy.org/doc/stable/reference/index.html).
3. **Check Existing Code:** Use `grep_search` to find if the function exists, is aliased, or is implemented in a
   submodule (like `cupy.linalg`).
4. **TDD approach:**
    - Create a test file in `tests/cupy_tests/` (e.g., `linalg_tests/test_vecdot.py`).
    - Use `@testing.numpy_cupy_allclose` to define expected behavior against NumPy.
    - Run the test to confirm it fails as expected.
5. **Implementation Patterns:**
    - **Wrappers/Aliases:** Use for simple renames or functions with added restrictions (like `cumulative_sum`).
    - **GUFuncs:** Use `cupy._core._gufuncs._GUFunc` for functions supporting broadcasting over specific axes (like
      `vecdot`).
    - **Namespace Export:** Ensure the function is exported in `cupy/__init__.py` and relevant submodules (e.g.,
      `cupy/linalg/__init__.py`).
6. **Verification:**
    - Run the specific test using `pytest`.
    - Run linters and type checkers: `ruff check .`, `mypy .`.
7. **Documentation:**
   - Make sure the function is documented in the appropriate submodule
   - Keep `numpy_cupy_implementation.md` updated with technical patterns and lessons learned.
8. **PR:** Submit a PR with the changes.
    - Track the PR, make sure auto linters, formatters, and tests pass.

## CI Stability and Best Practices

To prevent common CI failures observed in this project, adhere to the following:

### 1. Docstrings and Escape Sequences
*   **Always use Raw Strings:** Use `r"""..."""` for all docstrings. Python 3.12+ emits `SyntaxWarning` for invalid escape sequences (like `\*` or `\d`) in standard strings.
*   **Sphinx Strictness:** The CI runs Sphinx with `-p` (parallel) and `-W` (warnings-as-errors). A `SyntaxWarning` during import will crash the documentation build.
*   **Validation:** Run `ruff check .` locally before pushing. It will catch `W605` (invalid-escape-sequence) errors.

### 2. Symlink Integrity (Windows Development)
*   **Symlink Corruption:** If developing on Windows, ensure your Git configuration handles symlinks correctly to avoid replacing them with large source code blobs.
*   **Check Git Config:** Ensure `git config core.symlinks true` is set (requires Windows Developer Mode).
*   **Avoid Overwrites:** Never replace a symlink file (mode `120000`) with the actual content of the target file. If a symlink is "broken" or showing as a text file containing a path, do not paste the source code into it.
*   **Validation:** If a Git checkout fails with `File name too long`, check for symlinks that have been accidentally overwritten with source code using `git ls-tree -r HEAD`.