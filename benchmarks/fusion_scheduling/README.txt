Iterative fusion scheduling, 2026-11-09

Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition.
Environment: CPython 3.14.7, checkout's CuPy native artifacts.
Raw trial timings, source hashes and schedules are in results.json.

The benchmark alternates method order across seven trials, with twenty calls
per trial after warmup. CPU time measures submission; CUDA-event elapsed time
includes submission gaps and normal output/workspace allocation. Compilation,
input creation and numerical validation are excluded. The one-block control
uses the same generated code and wrapper, with its grid forced to one block.
These timings isolate scheduling; they are not comparisons against eager CuPy.

Median whole-call elapsed time (microseconds):

Function             Elements      One block    Scheduled
filter_agg               4,096          5.08         5.13
filter_agg              65,536         51.50         8.43
filter_agg           1,048,576        802.04         8.60
filter_agg           8,388,608      6,411.76        41.48
sigma_clipped_mean       4,096         22.91        22.96
sigma_clipped_mean      65,536        894.38       138.33
sigma_clipped_mean   1,048,576     14,826.02       189.75
sigma_clipped_mean   8,388,608    118,491.17       871.47

The 4,096-element calls select the ordinary one-block launch. Large calls use
up to 188 cooperative blocks, one per SM. Sigma clipping keeps three boolean
arrays; reduction scratch adds 1,504 bytes at 188 blocks. Grid synchronization
is used only for reductions. The crossover remains an initial heuristic, not
a hardware-independent optimum.

score() warm CPU submission time, before / current (microseconds):
64 x 64:       16.67 / 16.55
4096 x 64:     16.08 / 15.78
32768 x 512:   15.71 / 15.61

Both score frontends were measured in the same process with alternating trial
order. Their source hashes are identical: scheduling made no change to the
frontend or cached straight-line dispatch. score retains its three existing
regions and never enters the iterative scheduler. Small timing differences
should be interpreted as measurement variation.

Reproduce from the built checkout:
  python -m examples.fusion.bytecode_iteration_benchmarks --json /tmp/run.json
For before/after score measurements, add:
  --baseline-frontend /path/to/saved/_fusion_bytecode.py
The recorded run used --repeat 20 --trials 7. To reproduce the identical-code
score control after this change, use the current _fusion_bytecode.py as the
baseline path; its source is unchanged from the saved pre-scheduling frontend.

Validation: 106 new/existing fusion tests passed on CUDA. Graph capture checks
one kernel node for both functions at small and large sizes. Eleven selected
graph, schedule-transition and mask-update tests passed Compute Sanitizer's
synccheck with zero errors; see synccheck.txt. A focused 65,537-element probe of
both cooperative kernels passed memcheck with zero errors; see memcheck.txt.
The broader instrumented memcheck suite exceeded a 60-second time limit.

Memory-check probe, from the checkout:
  PYTHONPATH=. compute-sanitizer --tool memcheck --report-api-errors explicit \
    --error-exitcode 1 python benchmarks/fusion_scheduling/memory_probe.py
The explicit API-error mode excludes this driver's extended IMEX capability
warning; CUDA memory-access errors and explicit API failures remain checked.
