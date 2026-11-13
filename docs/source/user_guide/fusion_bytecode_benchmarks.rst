Bytecode fusion benchmark suite
===============================

The extended suite starts with score overhead and throughput, then measures
nested calls, changing scalars, boolean/shape branches, conditional allocation,
allocation between maps, metadata-derived shapes, nested instance methods,
array/scalar attributes, and multiple outputs. Numerical code and handwritten
``out=`` reference implementations live together in
``examples/fusion/_bytecode_benchmark_cases.py``. The runner is
``examples/fusion/bytecode_benchmarks.py``.

The saved run is in ``benchmarks/fusion_bytecode/results.json`` and
``results.csv``. ``results.rst`` contains generated tables. It covers 18 named
workloads, nine size/layout profiles, and 92 case/profile combinations:
335 measured configurations passed validation; 58 unsupported existing-fuser
configurations are recorded separately. The prototype implementation was not
changed for these benchmarks.

Comparators
-----------

Each case includes:

* ``eager``: the original expression, allocating ordinary temporary/results.
* ``eager_out``: every scratch/output buffer allocated before timing, with
  explicit ``out=`` operations. Constructor initialization still runs on each
  applicable call; buffers are overwritten before being consumed again.
* ``existing_fuse``: the existing ``Fusion`` wrapper, including its automatic
  ``new_fusion`` fallback. Known refusals are recorded with their exception;
  numerical mismatches are fatal, never classified as unsupported.
* ``prototype``: the private bytecode frontend, including its normal result
  allocation and diagnostics overhead.

Constructor cases additionally include ``eager_out_constants``: the handwritten
reuse implementation initializes persistent constant arrays outside timing.
This is a stronger numerical baseline, with different allocation/initialization
lifetimes. For example, allocation-mid requires an additional full-size buffer
to keep its constant ones array intact while scratch is reused. Persistent
scratch bytes are recorded for both reuse baselines. Reused outputs are
overwritten on subsequent calls, unlike normal independently owned outputs.

Neither reuse baseline substitutes a custom fused kernel, CUDA graph, or
algebraically simplified expression. Per-operation CuPy calls remain separate.
Initialization cannot simply be omitted from the ordinary reuse baseline:
some scratch buffers are overwritten by later operations in the same call.

Coverage
--------

.. list-table:: Input profiles, float32 by default
   :header-rows: 1

   * - Profile
     - Shape
     - Layout
   * - micro
     - 1 x 1
     - C contiguous
   * - tiny
     - 8 x 32
     - C contiguous
   * - small
     - 64 x 64
     - C contiguous
   * - medium
     - 4096 x 64
     - C contiguous
   * - large
     - 32768 x 512
     - C contiguous
   * - wide
     - 128 x 131072
     - C contiguous
   * - transposed
     - 32768 x 512
     - Transposed dense input
   * - strided
     - 32768 x 512
     - Every second column of a wider input
   * - xlarge
     - 131072 x 1024
     - C contiguous, 512 MiB of input data

Score covers every profile. Every available feature case covers small, large,
xlarge, transposed, and strided profiles. Shape-alternating uses two contiguous
views of the same input storage with widths 64 and 512 and equal element counts;
it is omitted from noncontiguous profiles. Other alternating cases cycle through
both boolean values, two instances of one class, or three scalar values. Python
and NumPy scalar sources are measured separately.

Method cases use the same class function definitions across instances and
reread their scalar/array attributes. The nested-chain expression calls two
ordinary Python helpers. Cache counts confirm one plan for changing numerical
scalars and instances, and two plans for alternating boolean/shape outcomes.

Measurement and validation
--------------------------

Inputs are resident on one GPU and use one nonblocking stream. Every argument
variant and every measured implementation is checked against allocating eager
CuPy before timing, including result shape, dtype, and numerical agreement
(``rtol=3e-5``, ``atol=2e-6``). Validation runs on the GPU and transfers only
scalar checks/error summaries to the host. No array copies to the host are
included in timing.

Warm measurements exclude input creation, scratch allocation, validation and
initial compilation. First observed calls, including device completion, are
recorded separately; disk/native kernel caches are not cleared, so these are
not guaranteed cold-toolchain measurements. All variants are warmed before
measurement, and prototype plan/function counts must remain stable during
timed batches.

The saved run uses five trials per method. Batch lengths adapt to pilot timings
with a 30 ms target and 3--300 calls, rounded up to balance argument variants.
Very slow existing-fuser cases can exceed the target with three calls. Method
order rotates between trials. JSON retains each trial, minima/maxima, standard
deviations, first-call timings, source hashes, environment and cache metadata.

CPU time measures average submission/dispatch time within a batch. Large calls
can encounter driver backpressure; use small inputs to assess host overhead.
CUDA events surround complete batches and include submission gaps, not just
kernel durations. Throughput counts logical input elements divided by event
elapsed time; it does not claim measured device bandwidth. Region counts are
not hardware kernel-launch counts.

The saved run used Python 3.14.7, NumPy 2.5.3, the checkout's existing
CuPy 14.3.0.dev0 native artifacts, CUDA runtime 13.2, and one NVIDIA RTX PRO 6000
Blackwell Workstation Edition. The existing artifacts emit a ``ParameterInfo
size changed`` ABI warning. The warning is retained in the report; native
modules were not rebuilt, limiting the scope of the validation. Representative
float64 cases were also numerically validated, without a full float64 timing
sweep.

Reproduction
------------

From the built checkout, with GPU access::

    python -m examples.fusion.bytecode_benchmarks \
        --profiles micro tiny small medium large wide transposed strided xlarge \
        --cases score --json /tmp/fusion-benchmarks.json
    python -m examples.fusion.bytecode_benchmarks \
        --profiles small large xlarge transposed strided \
        --resume --json /tmp/fusion-benchmarks.json
    python -m examples.fusion.bytecode_benchmark_report \
        /tmp/fusion-benchmarks.json

The runner checkpoints JSON/CSV after each complete case. Resume requires
matching implementation/source hashes, GPU name, and timing configuration.
The report generator checks data integrity and source hashes without importing
CuPy. List cases or select a smaller run with::

    python -m examples.fusion.bytecode_benchmarks --list-cases
    python -m examples.fusion.bytecode_benchmarks \
        --profiles large xlarge --cases nested_numpy_scalars allocation_mid \
        --json /tmp/fusion-selected.json

``--dtype float64``, ``--device``, ``--trials``, ``--target-ms``, and repeat
limits are also configurable. Use a separate output file when changing the
timing configuration or dtype.
