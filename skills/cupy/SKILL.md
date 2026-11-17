---
name: cupy
description: >-
  Use for any task or question involving CuPy: writing, porting, speeding up,
  benchmarking, or debugging Python code that runs NumPy or SciPy array code on NVIDIA GPUs.
  Use when writing array or scientific code in Python and you have a GPU available
  to get GPU acceleration.
  Converting NumPy/SciPy scripts to cupy and cupyx.scipy (FFT, linalg, sparse,
  signal), moving arrays between CPU and GPU, CuPy code that is slower than
  expected or slower than NumPy, timing CuPy vs NumPy, custom CUDA kernels from
  Python (RawKernel, RawModule, ElementwiseKernel, cupyx.jit) that crash or return wrong
  results, and CuPy with threads, multiprocessing, multiple GPUs, or containers.
license: CC-BY-4.0
metadata:
  author: NVIDIA
  tags:
    - cupy
    - numpy
    - scipy
    - gpu
---

# CuPy

CuPy is a NumPy/SciPy-compatible array library for NVIDIA GPUs. Most NumPy
code ports by swapping the import, so the work is less about the API and more about
GPU best practices: keeping data on the GPU, avoiding hidden synchronization,
and measuring correctly.

## Installation

Install a prebuilt wheel matching the CUDA major version: `pip install cupy-cuda12x` or
`pip install cupy-cuda13x`. Add `[ctk]` (`pip install "cupy-cuda13x[ctk]"`) when no CUDA
Toolkit is installed. Plain `pip install cupy` builds from
source and takes a long time. Install only one CuPy package per environment.
For conda: `conda install -c conda-forge cupy`.

Check the environment before answering version-specific questions:
`python -c "import cupy; cupy.show_config()"`. `cupy.cuda.is_available()`.

CuPy v14 requires CUDA 12+, NumPy 2, and follows NumPy 2 type promotion.
`cupy.sparse`, `cupy.prof` were removed: use `cupyx.scipy.sparse`, `cupyx.profiler`.
cuDNN support was removed.

SciPy equivalents live in `cupyx.scipy.` (`fft`, `linalg`, `sparse`, `signal`, and more). Coverage is partial: confirm a
function exists in the installed version (`hasattr(cupyx.scipy.signal, "welch")`)
before using it, instead of assuming.

APIs that look like they should exist but do not (CuPy 14):

| Missing | Use instead |
|---|---|
| `cp.random.default_rng()` methods `normal`, `choice`, `permutation`, `shuffle`, `multivariate_normal` | `rng.standard_normal(...) * sigma + mu`, `rng.integers`, `rng.random` + `cp.searchsorted(cp.cumsum(p), u)` for weighted choice, `cp.argsort(rng.random(n))` for permutations; or the legacy `cp.random.normal/choice/permutation` |
| `cupyx.scipy.optimize`, `cupyx.scipy.integrate`, `cupyx.scipy.cluster` | Keep the solver on the CPU and evaluate the objective on the GPU, or vectorize (e.g. bisection over a batch) |
| `cupyx.scipy.stats.norm`, `pearsonr`, `linregress`, `gaussian_kde` | `cupyx.scipy.special.ndtr` / `ndtri` (normal CDF / inverse CDF), `cp.corrcoef`, closed-form least squares |
| `cupyx.scipy.interpolate.griddata` | `cupyx.scipy.interpolate.LinearNDInterpolator(points, values)(xi)` (GPU Delaunay, same result as `method="linear"`), `NearestNDInterpolator`, `CloughTocher2DInterpolator` |
| `scipy.spatial.cKDTree`, `ConvexHull` | `cupyx.scipy.spatial.KDTree`, `Delaunay` |

The opposite mistake is as common: assuming a function is missing and hand-writing it or
moving the step to the CPU. These exist and run on the GPU: `cupyx.scipy.linalg.expm`,
`cp.linalg.eig` / `eigvals`, `cupyx.scipy.spatial.KDTree` and `Delaunay`,
`cupyx.scipy.interpolate.LinearNDInterpolator`, `cupyx.scipy.ndimage.label` and
`distance_transform_edt`. Run `dir()` on the `cupyx.scipy` module first.

## Porting NumPy code

1. Replace `import numpy as np` with `import cupy as cp` for the array math (keep NumPy
   for I/O and CPU-only libraries). `cupyx.scipy` replaces `scipy`.
2. Move data to the GPU once, at the start: `x = cp.asarray(x_np)`. Generate data directly
   on the GPU when possible, in the precision you need
   (`cp.random.default_rng(0).standard_normal(n, dtype=cp.float32)`).
3. Move results back once, at the end or when needed: `x.get()` or `cp.asnumpy(x)`.
4. NumPy ufuncs on CuPy arrays dispatch to CuPy. Passing CuPy arrays into CPU-only libraries
   (matplotlib, pandas, sklearn) needs `.get()` first.
5. Validate against the NumPy version on a small input with `cp.testing.assert_allclose` /
   `np.testing.assert_allclose(gpu.get(), cpu, rtol=...)`. Results can differ in the last
   bits (different reduction order, float32 vs float64), so compare with tolerances.

Behavior that differs from NumPy and can silently change results (out-of-bounds indices
wrap instead of raising, duplicate-index assignment is undefined, reductions return 0-d
arrays, float-to-int casts, random streams differ for the same seed): see
https://docs.cupy.dev/en/stable/user_guide/difference.html.

## Performance best practices

1. Keep data on the GPU. Every `cp.asarray`/`.get()` crosses PCIe. A pipeline that
   bounces between NumPy and CuPy is usually slower than pure NumPy.
2. No Python loops over elements. `for i in range(n): x[i] ...` launches one kernel
   (and often one sync) per element. Vectorize, or write a custom kernel.
   The same applies to many small problems: stack them into one batched array
   (`cp.linalg.inv`, `solve`, `matmul`, `fft` all accept a leading batch dimension).
   Loops of independent iterations (bootstrap resamples, Monte Carlo trials, parameter
   sweeps) should become an extra array dimension too: draw all indices at once
   (`idx = rng.integers(0, n, size=(n_boot, m))`), gather with `x[idx]`, and reduce along
   an axis, chunking if memory is tight.
3. Watch for hidden synchronization. These block until the GPU finishes and copy to
   host: `.get()`, `float(x)`, `int(x)`, `x.item()`, `print(x)`, `if x.any():`,
   `while err > tol:` on a 0-d array, and boolean-mask indexing / `cp.nonzero` (output size
   is unknown). In iterative solvers, a per-iteration `if err < tol:` check can
   hurt the performance greatly. Fix it by computing the error and checking it only every
   N iterations (e.g. 50-100):

   ```python
   for it in range(max_iter):
       u_new = step(u)
       if it % 50 == 49:                  # one reduction + one sync per 50 steps
           if float(cp.abs(u_new - u).max()) < tol:
               u = u_new
               break
       u = u_new
   ```

   This can run up to N-1 extra iterations past the CPU version's stopping point, which
   only moves the solution further toward convergence. Validate against the CPU result
   with a tolerance, not exact equality, and mention the different iteration count.
4. Size matters. Each kernel launch costs many µs, so NumPy usually wins
   below ~10^4-10^5 elements per operation. Batch small problems into one large array
   instead of looping over them, and benchmark at the real problem size.
5. Choose precision deliberately. Consumer and workstation GPUs (GeForce, RTX) run
   float64 at a small fraction of float32 speed. Use float32 when accuracy allows;
   datacenter GPUs (A100, H100, B200) have full-rate float64. For float32 matmul-heavy code,
   `CUPY_TF32=1` (set before CuPy initializes, no code changes) lets cuBLAS use TF32 Tensor
   Cores on Ampere and newer GPUs, including GeForce RTX 30/40: 1.5x on an 8192x8192 matmul
   on an RTX 4090. TF32 keeps float32's range but rounds the inputs to a 10-bit mantissa
   (float32 has 23), so expect about 1e-3 relative error per product. Accumulation stays
   in float32, so measured matmul error is often lower. Explain this trade-off and
   validate against a float32 result before enabling it.
6. Pre-allocate and reuse with `out=` arguments in hot loops
7. Fuse elementwise chains. Each CuPy operation is a separate kernel that writes a full-size
   temporary, so long expressions are memory-bandwidth bound. Wrapping the function in
   `@cp.fuse()` (or writing one `cp.ElementwiseKernel`) runs it as a single kernel: a
   10-operation float32 activation on 64M elements went from 12.8 ms to 0.6 ms.
   Keep whole-array reductions out of the fused code: a plain `.sum()`/`.max()` uses CUB,
   while the same reduction as a `cp.ReductionKernel` or inside `cp.fuse` was 20x slower
   (`((x - y) ** 2).sum()` on 134M float32: 3.5 ms plain, 69 ms as a ReductionKernel).
8. Small arrays with many steps are launch-bound (GPU looks idle in `nvidia-smi`). Fuse each
   step into as few kernels as possible, then capture the loop in a CUDA graph so it is
   replayed without Python or launch overhead (5.5x on 20,000 steps of a 192x192 stencil):

   ```python
   s = cp.cuda.Stream(non_blocking=True)
   with s:
       step(u, v)                 # warm up first: kernels must be compiled before capture
       s.begin_capture()
       for _ in range(100):
           step(u, v)             # update u, v in place; no host syncs or new inputs inside
       graph = s.end_capture()
       for _ in range(n_steps // 100):
           graph.launch(s)
       s.synchronize()
   ```

## Measuring performance

GPU calls are asynchronous: `time.time()` around a CuPy call measures the launch, not the
work. Use the built-in profiler, which warms up and synchronizes:

```python
from cupyx.profiler import benchmark

print(benchmark(my_func, (x,), n_repeat=20, n_warmup=3))   # reports CPU and GPU times
```

Or synchronize manually: `cp.cuda.Device().synchronize()` before stopping the timer. The
first call of each kernel compiles it (cached in `~/.cupy/kernel_cache`), so always warm up
and exclude the first run. Compare against NumPy at the real problem size, and include
transfer time if the real pipeline has it.

## Memory

- CuPy keeps freed memory in a pool, so `nvidia-smi` shows it as used.
  `cp.get_default_memory_pool().used_bytes()` / `.total_bytes()` show the real usage;
  `.free_all_blocks()` returns cached blocks to the driver (needed before handing memory
  to another library such as PyTorch).
- Limit CuPy share of the GPU with `CUPY_GPU_MEMORY_LIMIT="50%"` (or bytes), or
  `cp.get_default_memory_pool().set_limit(size=...)`.
- If you get Out of memory errors try to: process in chunks, delete large temporaries (`del x`),
  use float32, and free FFT plan caches (`cp.fft.config.get_plan_cache().clear()`) if many FFT shapes are used.
- Large host-to-device transfers are faster from pinned memory: allocate host buffers with
  `cupyx.empty_pinned(shape, dtype)` and copy with `cp.asarray` or `x.set(...)`.
- Overlap transfers and compute with non-default streams:
  `with cp.cuda.Stream(non_blocking=True) as s: ...; s.synchronize()`.

## Custom kernels

Use them when vectorized CuPy and `cp.fuse` are not enough: `cp.ElementwiseKernel` and
`cp.ReductionKernel` for C snippets, `cp.RawKernel` / `cp.RawModule` for full CUDA C++,
`cupyx.jit.rawkernel` for Python-syntax kernels (experimental). See
https://docs.cupy.dev/en/stable/user_guide/kernel.html before writing or debugging one.
RawKernel does no argument checking, so the common bugs are silent wrong results:

- Python `int`/`float` scalars are passed as `long long`/`double`; convert to the C type
  (`np.float32(a)`, `np.int32(n)`, `x.dtype.type(v)`).
- The array dtype must match the pointer type (float64 data into `float*` is garbage).
- Inputs must be C-contiguous (`cp.ascontiguousarray`); views like `a[:, 3]` or `a.T` are not.
- In float kernels write `1.0f`, `expf`, `sqrtf`: a double literal promotes the math to
  float64 (71 ms vs 1.0 ms for the same kernel on an RTX 4090).

Kernel errors (`cudaErrorIllegalAddress`) are reported at a later, unrelated call and
leave the CUDA context unusable: restart the process, then localize with
`CUDA_LAUNCH_BLOCKING=1` and `compute-sanitizer python script.py`.

## Threads, processes, and deployment

- The current device is thread-local: `cp.cuda.Device(1).use()` in the main thread does not
  apply to worker threads; select the device inside each worker (`with cp.cuda.Device(i):`).
- Threads share the default (null) stream, which serializes their GPU work. Give each thread
  its own `cp.cuda.Stream(non_blocking=True)`, or set `CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1`.
- A process forked after CUDA is initialized (any CuPy call) cannot use the GPU
  (`cudaErrorInitializationError`). Use the `spawn` start method, and pin workers to GPUs
  with `CUDA_VISIBLE_DEVICES` or `cp.cuda.Device(i)`.
- Kernels are compiled on first use and cached in `~/.cupy/kernel_cache` (keyed by source,
  CuPy/CUDA version, and GPU architecture). In containers set `CUPY_CACHE_DIR` to a writable
  or pre-warmed directory, warm up before serving, or use `CUPY_CACHE_IN_MEMORY=1`.

## Going further
- Interop for zero-copy with other libraries: DLPack: `cp.from_dlpack(t)` / `torch.from_dlpack(x)` with PyTorch and JAX;
  `__cuda_array_interface__` with Numba and others; cuDF `Series.to_cupy()` /
  `cudf.Series(cupy_array)`. Data stays on the GPU, so make sure the producer's stream work
  is finished or on the same stream.
- Multiple GPUs: `with cp.cuda.Device(1): ...` places arrays and kernels on device 1;
  arrays live on the device where they were created.

## External documentation

- User guide: https://docs.cupy.dev/en/stable/user_guide/index.html
- Differences from NumPy: https://docs.cupy.dev/en/stable/user_guide/difference.html
- Custom kernels (Elementwise, Reduction, Raw, JIT): https://docs.cupy.dev/en/stable/user_guide/kernel.html
- `cupyx.jit` API: https://docs.cupy.dev/en/stable/reference/kernel.html
- Performance tips: https://docs.cupy.dev/en/stable/user_guide/performance.html
- API reference and NumPy/SciPy coverage: https://docs.cupy.dev/en/stable/reference/
- If you are able to clone repos you can clone it into a temp location to verify APIs:
  https://github.com/cupy/cupy
- Upgrade notes: https://docs.cupy.dev/en/stable/upgrade.html
