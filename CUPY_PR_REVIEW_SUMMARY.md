# CuPy Pull Request Review Insights

This document summarizes project-specific patterns and "unwritten rules" distilled from past closed pull requests and code reviews in the CuPy repository.

## 🚀 Performance & Synchronization

### 1. Avoid Device-to-Host (DtoH) Synchronization
*   **The Cardinal Sin:** Any logic that performs a GPU check (e.g., `cupy.all(indices[:-1] <= indices[1:])`) and then branches on the result in Python is heavily scrutinized. This forces a synchronization that kills pipeline performance.
*   **The Preference:** Stay on the device as long as possible. If a check is needed, consider if it can be part of the kernel or if the "slow path" can just handle the edge case without a check.

### 2. Layout Awareness (C vs. F Contiguity)
*   **Priority:** Kernels and high-level APIs are expected to handle both C-contiguous and F-contiguous (Fortran) memory layouts correctly.
*   **Bug Status:** Layout-specific bugs are treated as high-priority "silent failures" because they often produce incorrect results without crashing.

### 3. Hardware Atomicity & Complex Types
*   **Torn Reads/Writes:** Avoid marking 16-byte types (like `complex128`) as "primitive" in `NumericTraits`. Doing so can lead to torn reads/writes in advanced algorithms (like CUB's decoupled lookback scan) because 16-byte stores are not always atomic on all hardware.
*   **Correctness Over Speed:** Prioritize data integrity. If a fast path (like a CUB reduction) doesn't safely support a type, fallback to a slower but atomic-safe generic path.

### 4. GPU Hardware Limits
*   **Grid/Block Dimensions:** Reviewers are vigilant about potential overflows in kernel launches (e.g., `gridDim.y` overflow in sparse SpMM). Kernels must account for hardware limits on dimension sizes and provide workarounds (like grid-stride loops or tiling) when necessary.

### 5. Internal Synchronization Tools
*   **Memory Pooling:** Use project-specific synchronization tools like `pymutex` (found in `cupy.cuda.memory`) when implementing or modifying memory pools to ensure thread safety without reinventing synchronization primitives.

## 🧹 Resource Management & Memory

### 1. Robust Destruction (`weakref.finalize`)
*   **The Standard:** Use `weakref.finalize` for objects managing low-level resources (e.g., `curand` generators, stream handles) to avoid `ModuleNotFoundError` during interpreter shutdown.

### 2. Initialization Performance (`empty` vs `zeros`)
*   **Preference:** Use `cupy.empty` instead of `cupy.zeros` if the output array is guaranteed to be fully overwritten. Redundant zeroing is seen as unnecessary overhead.

### 3. Allocation Minimization
*   **Intermediate Reduction:** Algorithms that reduce working memory from $O(kN)$ to $O(N)$ (e.g., combining multiple axis shifts into a single `cupy.roll`) are highly preferred to prevent OOM errors on large arrays.

## 🏗️ Module Architecture & API

### 1. Lean Import Strategy (Lazy Loading)
*   **Rule:** Submodules like `cupyx` should be imported lazily (inside functions) rather than at the top level of core `cupy` modules.
*   **Goal:** Keep `import cupy` fast. The ~40ns overhead of a function-local import is negligible compared to GPU kernel execution.

### 2. SciPy Alignment & Deprecation
*   **Sync:** CuPy mirrors SciPy's deprecation timelines. Implement the replacement API first, then issue a `DeprecationWarning` for the old one.
*   **Submodules:** Use git submodules (like `xsf`) for sharing C++ logic with SciPy rather than manual vendoring.

## 🛠️ Cython, JIT & C++ Implementation

### 1. Cython as a Dispatcher
*   **Pattern:** `.pyx` files should primarily serve as a glue layer to dispatch between optimized "fast paths" (CUB, CUTLASS, cuBLAS) and generic "fallback paths."

### 2. Fast-Path Dispatching
*   **Bypassing Overhead:** For critical operations (like `matmul`), use `try_fastcall` patterns to bypass heavy `GUFunc` machinery for small matrices where Python/Cython overhead dominates.

### 3. Computational Precision
*   **Promotion Avoidance:** Avoid automatic promotion to `float64` in kernels. Use `float32` for `float32` and 8/16-bit inputs.
*   **Testing:** Use dtype-dependent tolerances in tests (e.g., `rtol={"default": 1e-7, numpy.float32: 1e-5}`) to ensure high precision for `float64` while allowing for `float32` variance.

## 📊 Sparse Matrix & Distributed Logic

### 1. Format Conversion Bottlenecks
*   **The Rule:** Avoid CSR to CSC (or vice versa) conversions. Implement kernels that handle the user's provided format directly.

### 2. Distributed Sparse Collectives
*   **Two-Phase Exchange:** Always use a two-phase pattern for sparse collectives: exchange metadata (sizes) first, then exchange the data. Interleaving them can cause hangs in NCCL.

## ⚡ Concurrency & Hardware Features

### 1. Per-Thread Default Stream (PTDS)
*   **Support:** Actively support PTDS for multi-threaded performance. This often requires specialized handling in the backend for both CUDA and ROCm.

### 2. ROCm/HIP Pragmatism
*   **Vigilance:** New hardware features or CI updates for ROCm (like ROCm 7.0 support) require coordinated effort in `cupy_builder` and kernel traits.

## 🎲 Random Number Generation (RNG)

### 1. Bijection Sampling
*   **The Standard:** Use cryptographic bijections (e.g., Feistel networks) for `replace=False` sampling to generate indices on-the-fly and avoid materializing the entire population array.

## 🧪 Testing & CI Best Practices

### 1. Comprehensive Dtype Coverage
*   **Explicit Testing:** Don't rely on generic tests; explicitly include `'D'` (`complex128`) and `'F'` (`complex64`).

### 2. Testing at Scale
*   **Regression:** Verify optimizations with very large arrays (up to $10^9$ elements) to ensure indexing doesn't overflow.
