Bytecode fusion benchmark results
=================================

18 named workloads, 9 input profiles, 92 case/profile combinations. 335
measured configurations passed validation; 58 existing-fuse refusals are
recorded.

GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition. Python 3.14.7, CuPy
14.3.0.dev0, NumPy 2.5.3, CUDA runtime 13020. Run started
2026-11-06T17:48:35.694298+00:00.

Each number is the median of 5 warmed batches. Batch lengths are adaptive and
balanced across argument variants. GPU elapsed time includes host submission
gaps. Gelem/s counts logical input elements, not device memory traffic. Full
trials, first-call times, dispersion, cache metadata, source hashes and buffer
sizes are in results.json; results.csv contains the flat measurements.

.. csv-table:: Score CPU submission time (microseconds)
   :header-rows: 1

   Profile,Eager,Reuse,Existing fuse,Prototype,Prototype / existing
   micro,58.65,55.87,11.27,15.43,1.37
   tiny,60.49,56.74,10.82,15.50,1.43
   small,59.22,56.50,10.89,15.71,1.44
   medium,59.94,56.45,11.20,15.74,1.41

.. csv-table:: Score complete-call elapsed time (microseconds)
   :header-rows: 1

   Profile,Input elements,Eager,Reuse,Prototype,Prototype Gelem/s,Speedup / reuse
   micro,1,58.66,55.87,15.43,6.48e-05,3.62
   tiny,256,60.49,56.75,15.50,0.0165,3.66
   small,"4,096",59.23,56.51,15.71,0.261,3.60
   medium,"262,144",59.95,56.45,15.74,16.7,3.59
   large,"16,777,216",685.38,645.64,54.47,308,11.85
   wide,"16,777,216",698.29,647.09,124.50,135,5.20
   transposed,"16,777,216",826.00,793.76,57.39,292,13.83
   strided,"16,777,216",815.95,781.92,174.33,96.2,4.49
   xlarge,"134,217,728",8129.92,8151.34,659.37,204,12.36

.. csv-table:: Largest contiguous inputs: complete-call milliseconds
   :header-rows: 1

   Case,Eager,Reuse + initialization,Reuse + constant initialization hoisted,Prototype,Speedup / best reuse
   score,8.130,8.151,-,0.659,12.36
   row_norm,1.017,1.018,-,0.328,3.11
   map_chain,5.014,5.019,-,0.727,6.91
   nested_numpy_scalars,7.918,7.936,-,0.727,10.92
   nested_python_scalars,7.923,7.945,-,0.727,10.93
   boolean_true,4.294,4.291,-,0.729,5.89
   allocation_true,5.827,5.823,5.350,1.532,3.49
   boolean_false,3.235,3.239,-,0.729,4.44
   allocation_false,3.235,3.240,3.241,0.728,4.45
   boolean_alternating,3.763,3.764,-,0.728,5.17
   allocation_alternating,4.529,4.529,4.296,1.129,3.80
   shape_branch,3.958,3.967,-,0.728,5.45
   shape_alternating,3.770,3.770,-,0.729,5.17
   allocation_mid,5.828,5.821,5.357,2.587,2.07
   shaped_allocation,4.272,4.259,4.267,0.736,5.78
   instance_method,4.694,4.700,-,0.729,6.45
   instance_alternating,4.689,4.695,-,0.728,6.45
   multiple_outputs,4.289,4.293,-,1.140,3.77

The constant baseline initializes persistent constant arrays outside timing.
Ordinary reuse still initializes them per call. Both reuse baselines overwrite
returned arrays on later calls. The constant baseline can retain additional
full-size buffers. These are numerical comparisons, not equivalent allocation
lifetimes or observable allocation effects.

.. csv-table:: Cache reuse on changing inputs
   :header-rows: 1

   Case,Argument variants,Cached plans,Analysed definitions
   nested_numpy_scalars,3,1,3
   nested_python_scalars,3,1,3
   boolean_alternating,2,2,1
   allocation_alternating,2,2,1
   shape_alternating,2,2,2
   instance_alternating,2,1,2

.. csv-table:: Speedup over best reuse on noncontiguous inputs
   :header-rows: 1

   Profile,Minimum,Case,Maximum,Case
   transposed,1.23,allocation_mid,13.83,score
   strided,1.52,row_norm,7.57,nested_python_scalars

Existing fuse includes its automatic new_fusion fallback. Its multi-stage
schedules can be particularly slow on these reductions; large speedups over
that path are workload-specific. Complete existing-fuse timings are preserved
in JSON/CSV. Small-input CPU overhead and large-input throughput should be
assessed separately.

The existing native build emitted these import warnings; the artifacts were not
rebuilt for this run::

    cupy._core._kernel.ParameterInfo size changed, may indicate binary incompatibility. Expected 48 from C header, got 64 from PyObject
