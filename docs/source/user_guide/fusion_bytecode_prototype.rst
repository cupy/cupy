Bytecode fusion prototype
=========================

An extended benchmark suite now compares score and the added features against
allocating eager CuPy and explicit ``out=`` reuse, including large and
noncontiguous inputs. See ``fusion_bytecode_benchmarks.rst`` and the saved
tables in ``benchmarks/fusion_bytecode/results.rst``.

The private ``cupy._core._fusion_bytecode`` module implements the first
prototype from ``fusion_compiler_plan.txt``. The public ``cupy.fuse`` API is
unchanged. No native extension rebuild or external compiler is required.

Usage
-----

From a built CuPy checkout::

    import cupy as cp
    from cupy._core._fusion_bytecode import fuse

    @fuse
    def score(x):
        energy = (cp.exp(-abs(x)) * cp.sin(x)**2).sum(axis=1)
        mass = (1 / (1 + cp.exp(-x))).sum(axis=1)
        return cp.sqrt(energy) / (mass + 1)

    x = cp.ones((4096, 64), dtype=cp.float32)
    report = score.explain(x)  # No user execution, allocation or CUDA compilation
    result = score(x)
    print(score.diagnostics)
    score.clear_cache()

Only ``fuse(function)`` and ``@fuse`` are currently provided. Diagnostics are
JSON-serializable and include generated verified Python source, operations,
callees, branch guards, allocation locations, analysis time, selected backends,
and failures. ``explain`` can populate analysis/plan caches but never calls the
user function or runs its operations. CUDA compilation remains lazy. Kernel
launch counts are explicitly unknown; a compiled region is not a launch count.
Phase-specific CUDA compilation timing is not yet implemented.

Branches, allocation, and Python callees
---------------------------------------

This example calls an ordinary, undecorated Python helper. Its bytecode is
analysed recursively and its arithmetic is inlined into the caller's region::

    def wave(x, bias):
        return cp.sin(x + bias) * cp.cos(x)

    def adjusted(x, use_bias):
        if x.shape[0] <= 8:
            return cp.tanh(x)
        if use_bias:
            bias = cp.ones_like(x)
            return wave(x, bias)
        return wave(x, 0)

    compiled = fuse(adjusted)
    x = cp.linspace(-1, 1, 160_000_000, dtype=cp.float32)

    a = compiled(x, True)      # Allocate bias, then execute one fused region
    b = compiled(x, False)     # Inline wave without allocating bias
    c = compiled(x[:4], True)  # Select tanh; neither allocate nor analyse wave

    In []: %timeit adjusted(x, False)
    3.86 ms ± 69 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

    In []: %timeit compiled(x, False)
    869 μs ± 20.4 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)


The helper is defined outside ``adjusted``: this demonstrates a nested call,
not runtime function creation inside the compiled body. Function creation is
not currently supported. Repeated analysis of ``wave`` reuses its identity
cache entry. Selected boolean/shape paths have separate guarded plans, and
allocation still happens on every execution of the applicable path.

The existing tracing-based ``cp.fuse(adjusted)`` rejects the shape lookup with
``NotImplementedError: `shape` is not supported, currently.`` Calling a Python
helper alone is not new; the added functionality here is its composition with
shape-based branch selection and conditional host allocation.

Architecture
------------

The frontend follows a selected path through ``dis`` instructions, representing
numerical work as operation/value nodes. It inlines supported Python callees
without executing them. Operation classes own identity matching, accepted
keywords, and implementation emission. Only registered CuPy operations enter
the generated region functions; arbitrary Python callbacks are rejected.

The executor is also a generated Python function, so warm calls do not replay
the bytecode interpreter or execute a general instruction dispatch loop.
Allocations are explicit host calls between numerical regions. Each numerical
region is compiled through the legacy ``_FusionHistory`` code generator, with
recognized compilation refusals selecting ``new_fusion`` before launch.
Runtime errors propagate without replaying the region or original function.
Fusion thread-local state is restored after legacy analysis.

For multiple reductions, the initial partitioner splits after each reduction.
This avoids the grid-synchronized multi-reduction path for the score example.
It is a simple heuristic, not a scheduling cost model. Single-reduction
functions can retain their post-map in the same region. Straight-line regions
reuse the existing CUDA code generators and reductions; the iterative frontend
below has a small dedicated CUDA emitter.

Literal squares and fallback scheduling
--------------------------------------

Array powers with a literal Python integer exponent of 2 use ``square``
instead of the general ``power`` ufunc. Boolean arrays retain the int64
promotion of ``ndarray.__pow__``. Runtime exponents are not specialized by
value. The device-loop emitter also lowers literal squares to multiplication.
Reduction dtype/rank inference lets this apply through reduced means; arrays
used in broadcasting, such as column means, are computed before higher-rank
consumers instead of being recomputed for every input element. A reduction
epilogue that changes dtype is kept separate when required by the CUB backend.

The tracing fallback backend sizes CUDA grids from the active device's SM
count and the compiled kernel's occupancy, with a cap for small input work.
When a large reduction has fewer output groups than SMs, its output-driven
fused reduction cannot use the whole device. That execution switches to
ordinary parallel CuPy reductions between fused elementwise stages. The same
staging path handles devices without cooperative launch support; it is also
used for multi-stage execution on HIP. Device limits are cached per device,
while the scheduling decision remains dependent on the current input shape.
Staged CUDA signatures distinguish ranks, contiguity and index widths.

This uses the existing reduction implementations and their dtype, axis and
stream handling rather than introducing a device-specific reduction tree.
It may use more than one CUDA kernel for a frontend region. See the actual
before/after timings in ``benchmarks/cupyx_fusion/fixes.rst`` and the raw
``fixed_results.json``; the earlier launch-grid changes in
``regressions.rst`` were diagnostic controls, whereas these are implemented
production paths. The native extension must be rebuilt for scheduling changes.

Supported subset
----------------

* Read-only numerical expressions using single-output CuPy namespace ufuncs,
  including inverse/hyperbolic trig, ``arctan2``, ``isnan``, ``isinf``,
  ``isfinite`` and logical operations, and ``sum``, ``prod``, ``min``, ``max``,
  ``all``, and ``any``. Ufunc identities are discovered once at import and
  dispatched through a dictionary; aliases share their canonical operation.
  Multi-output ufuncs such as ``modf`` and ``frexp`` remain unsupported.
  Reduction options must be static. Array reduction methods normalize to
  their registered CuPy operations.
* Statically resolved Python callees, including definitions without source.
  Recursive call cycles are rejected.
* Positional/keyword/default argument binding, fixed tuples, and ordinary
  instance methods declared with ``@fuse``. Method receivers conventionally
  use the first parameter name ``self``. Numerical instance attributes are
  read again on each execution; methods are analysed by their underlying
  class function definition, including overrides.
* Branches controlled by input booleans or supported input-array metadata
  predicates, including shape comparisons and dtype comparisons. The selected
  branch alone is analysed. Guard outcomes are checked at dispatch, allowing
  shape-dependent variants under the same dtype/ndim key.
* ``empty``, ``zeros``, ``ones``, ``full`` and their ``_like`` variants, using
  supported host shape arguments or input-array metadata. Allocations remain
  at runtime and inside their selected branch. Initialization is delegated to
  the original constructor rather than fused with its consumer.

NumPy arrays, numerical subclasses, properties, custom attribute hooks,
positional output arguments and ``out=`` remain unsupported. Exceptions,
generators, coroutines, recursive calls, dynamic call resolution and arbitrary
callbacks are explicitly excluded from the compiler's scope.
Metadata of computed intermediate arrays, variadic signatures and discarded
arithmetic regions are
also outside the subset. A scalar-only region invokes its registered CuPy
operations directly; it does not yet combine them into one kernel.
The frontend accepts CPython 3.11--3.14; runtime validation currently covers
3.14 only. It does not use CPython frame hooks.

Analysis-time dispatch and composition
--------------------------------------

The operation registry distinguishes fusible operations from ordered library
boundaries. The emitter flushes numerical regions at a boundary, invokes the
known implementation on the current stream, and resumes fusion afterwards.
The initial library set includes ``matmul``/``@``, ``dot``, ``tensordot``,
``reshape``, ``transpose``/``.T``, ``ascontiguousarray``, ``mean``, ``std``,
``linalg.solve``, ``linalg.inv``, ``linalg.cholesky``, ``linalg.norm`` and
one-dimensional ``fft.fft``/``fft.ifft``. This is an explicit registry, not
permission to call arbitrary functions found in the CuPy namespace.

Statically resolved numerical helpers are first analysed inline. If a helper
requires the iterative frontend, analysis prepares that kernel as a boundary
instead. For example, ``sigma_clipped_mean(cp.sin(data @ weight))`` composes a
matrix-library call, a fused linear region and a cooperative loop kernel.
Preparing such a helper uses dtype/rank metadata rather than evaluating CuPy
operations or allocating sample arrays. Initial computed-argument typing covers
registered elementwise operations, 1D/2D ``matmul``/``dot`` and basic indexing;
other combinations are rejected explicitly. Source-free functions still work.

Indexing, slice assignment and ``+=``/``-=``/``*=``/``/=`` on arrays are ordered
boundaries delegated to native CuPy. Their receiver must be C-contiguous.
Pending expressions are materialized before a write; views retain native
aliasing, overlapping assignments use CuPy's implementation, and errors
preserve preceding effects without replay. General indexed writes inside a
cooperative device loop are not implemented.

Short host-known ``range`` loops containing library operations can be expanded
during analysis on CPython 3.14, up to 32 total iterations. Input integer bounds
are guarded, so changing a bound selects another plan. This is bounded
specialization, not a general host-loop executor. Library calls in
device-dependent loops remain unsupported. Ordinary device loops continue to
use the single-kernel path described below. Dispatch occurs at analysis;
cached ``score`` calls acquire no extra runtime dispatcher. See
``examples/fusion/bytecode_dispatch.py`` for runnable compositions.

Single-kernel iteration prototype
--------------------------------

On CPython 3.14, the private ``fuse`` wrapper also tries a small CUDA bytecode
lowering when the straight-line frontend refuses a function. The target is one
scalar-output kernel for element counting and iterative sigma clipping, as
shown in ``examples/fusion/bytecode_iteration.py``. This path emits device jumps,
loops and reductions directly, rather than calling CuPy operations in a
host loop. Analysis does not execute the original function or compile CUDA.

An element loop can conditionally add independent contributions to a
zero-initialized scalar accumulator. Threads visit disjoint input elements and
combine their accumulators before returning. Reading the running accumulator
in a predicate, breaking out of element iteration, and returning an element
local are rejected because these depend on sequential iteration order.

Uniform ``range(stop)`` loops support conditions and ``break`` based on device
reductions. Boolean indexing of the input becomes a virtual filtered view:
``len``, ``mean`` and population ``std`` visit selected values without allocating
or compacting a variable-length array. Selection is snapshotted when a view is
stored, so subsequent mask updates do not change that view. ``all`` supplies
convergence tests. Masks and temporary selection storage are allocated before
launch and reused across iterations; initializing and updating them happens in
the same kernel. Sigma clipping uses three boolean workspace arrays.

This deliberately limited implementation accepts one one-dimensional input
array, Python scalar parameters, and produces a zero-dimensional
CuPy output. In particular, counting returns a device int64 scalar instead of
reading a Python integer back to the host. Integer, float32 and float64 inputs
and strided views are supported; float16 and complex inputs are excluded.
Array results, input mutation, general indexing and loop-carried element
dependencies remain unsupported inside this device kernel. Local input-array
rebinding with unchanged dtype uses reusable workspace, leaving the caller's
input unchanged; it cannot be combined with filtered views. Registered basic
arithmetic and ``sin``, ``cos``, ``tan``, ``exp``, ``log``, ``sqrt`` and ``tanh``
can occur inside device loops. The explicit device-math table also covers
inverse/hyperbolic trig, binary ``arctan2``/``hypot``, logarithm/exponential
variants, and ``isnan``/``isinf``/``isfinite``/``signbit``. This emitter uses
registered CUDA scalar math functions, rather than assuming every namespace
ufunc has a device-loop lowering. Reduction order can
change floating-point rounding and, near a clipping threshold, mask decisions.

Scheduling is confined to iterative plans; cached straight-line calls such as
``score`` do not invoke this scheduler. Inputs up to 16,384 elements use an
ordinary launch of one block of 256 threads. Larger inputs use a cooperative
grid when the device supports it, otherwise retaining the one-block schedule.
The initial heuristic assigns about 4,096 elements per block and caps the grid
at one block per SM, bounded by the compiled kernel's occupancy. Device
capabilities, occupancy and compiled variants are cached; input size remains
dynamic under the same dtype/ndim plan.

Each function body has a sequence of phases. Dense array passes and element
iteration retain the same grid-stride ownership, so an element is read by the
thread that wrote it and these phases need no barriers. Reductions first
combine thread values within each block, then publish block partials into
scratch storage and synchronize the grid. Every block computes the same
result before continuing. A second barrier protects scratch reuse. Counting
needs this collective only at the end; sigma clipping needs collectives inside
each iteration for statistics and convergence. All blocks take the same
reduction-dependent branches and loop exits. This preserves one kernel launch
without a host-controlled iteration loop or an initialization kernel.

Sigma clipping retains three boolean workspace arrays. Multi-block execution
also allocates eight bytes of reduction scratch per block, reused for all
reductions. Single-block calls need no reduction scratch or cooperative launch.
``explain`` includes generated CUDA, phase descriptions, workspace count and
the planned launch count. After execution, ``diagnostics['schedule']`` describes
the selected device, block count and reduction scratch size. CUDA graph tests
check the actual launch structure. The heuristic is intentionally simple;
cooperative launches and global barriers can cost more than one-block execution
near the crossover. See ``examples/fusion/bytecode_iteration_benchmarks.py``
for full-call timings against a forced one-block control and optional
before/after ``score`` frontend measurements.

Caching and scalar policy
-------------------------

Each wrapper owns an analysis cache keyed by ``id(function)``. Strong function
references prevent accidental reuse of an identity after garbage collection.
Stable function definitions reuse analysis; lambdas and dynamic functions
recreated on every call do not share entries merely because their code matches.

Plan selection uses numerical dtype/ndim, Python scalar origin, boolean values,
receiver class, and metadata-predicate outcomes. Numerical scalar values are
runtime inputs. The existing backends retain their own kernel/shape caches.
Changing an array's shape does not freeze an observed host allocation shape.

There are no automatic dependency guards for replaced code, defaults, global
or closure bindings, class definitions, or operation registrations. Call
``clear_cache()`` on each affected root wrapper after such changes. This clears
both its plans and its transitive callee analyses. Changing ordinary numerical
argument values or stored numerical attribute values does not require clearing;
changing an attribute's structural role does.

Python scalar input representations are int64, float64, complex128, and uint8
for bool. Input integer conversion can raise on values outside int64. Integer
constants inside a function are reduced to signed 64-bit representation.
Weak Python scalar origin is retained for backend promotion, while NumPy
scalars retain supported explicit dtypes. uint8 boolean storage also means
numerical operations on converted Python booleans use uint8 arithmetic.
Branch specialization retains the original boolean provenance.

Arithmetic follows the executing CUDA implementation. NumPy warnings and
``seterr`` callbacks are not emulated. No arithmetic is evaluated to discover
errors during analysis; CUDA may produce NaN/Inf instead of host exceptions.
No additional range checks emulate Python arbitrary-precision arithmetic.

Validation and measurements
---------------------------

On 2026-11-06, the prototype's 31 contract cases and four unchanged existing
fusion-cache cases passed together: **35 passed**. The checks cover numerical
results/dtypes, strides, tuple outputs, scalar conversion, source-free code,
callee caching, explicit invalidation, branch variants, conditional allocation,
method overrides, rejection without callbacks, and execution without replay.

The environment was CPython 3.14.7, NumPy 2.5.3, the existing checkout's
CuPy 14.3.0.dev0 native artifacts, and an NVIDIA RTX PRO 6000 Blackwell
Workstation Edition. GPU access required execution outside the sandbox.
The existing native artifacts emit a ``ParameterInfo size changed`` ABI
warning on import. They were not rebuilt for these checks; this limits the
validation claim. Other Python versions, GPUs, ROCm, and concurrent execution
have not been validated.

The following measurements are medians of three trials, each with 30 warmed
calls on one nonblocking stream. Inputs are resident float32 arrays. Every
method allocates its normal outputs; there is no ``out=`` preallocation.
Numerical results and dtypes are checked before timing. CUDA-event times
include host submission gaps and are not isolated kernel durations.
First-call timings in JSON may reuse disk compilation caches and are not
measurements of a completely cold toolchain.

.. list-table:: Complete-call CUDA-event elapsed time, microseconds
   :header-rows: 1

   * - Workload
     - Shape
     - Eager CuPy
     - Existing fuse
     - Prototype
   * - Score
     - 64 x 64
     - 63.77
     - 12.39
     - 16.93
   * - Score
     - 4096 x 64
     - 62.88
     - 460.87
     - 17.15
   * - Score
     - 32768 x 512
     - 681.44
     - 44461.72
     - 54.39
   * - Nested map
     - 32768 x 512
     - 181.71
     - 59.73
     - 60.78
   * - Allocation and boolean branch
     - 32768 x 512
     - 571.65
     - Unsupported
     - 143.32

Small-score CPU submission time is 16.89 microseconds versus 11.96 for existing
fusion (1.41x); event elapsed time is 1.37x. Both satisfy the prototype's 2x CPU
and 1.5x elapsed-time ceilings in this sample. The large score is about 12.5x
faster than eager and the allocation/branch example about 4.0x faster.
Existing fusion includes its automatic ``new_fusion`` fallback; its particularly
slow multi-reduction schedule explains the large score gap. These are local
workload measurements, not general performance guarantees.

Reproduce from the built checkout::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
        tests/cupy_tests/core_tests/test_fusion_bytecode.py \
        tests/cupy_tests/core_tests/fusion_tests/test_kernel_cache.py
    python -m examples.fusion.bytecode_prototype \
        --repeat 30 --trials 3 --json /tmp/cupy-bytecode-prototype.json

Plugin autoload was disabled because an unrelated local ``profile.py`` shadows
the standard library when the installed benchmark plugin imports ``cProfile``.
