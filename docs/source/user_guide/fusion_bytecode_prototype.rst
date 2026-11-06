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
functions can retain their post-map in the same region. There is no new CUDA
code generator, reduction implementation, or general optimization framework.

Supported subset
----------------

* Read-only numerical expressions using the registered arithmetic and math
  ufuncs, and ``sum``, ``prod``, ``min``, ``max``, ``all``, and ``any``.
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
opaque callbacks, mutations, positional output arguments, ``out=``, loops,
exception handlers, and device-data predicates are not supported yet.
Metadata of computed intermediate arrays, general array indexing, dynamic
call resolution, variadic signatures, and discarded arithmetic regions are
also outside the subset. A scalar-only region invokes its registered CuPy
operations directly; it does not yet combine them into one kernel.
The frontend accepts CPython 3.11--3.14; runtime validation currently covers
3.14 only. It does not use CPython frame hooks.

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
