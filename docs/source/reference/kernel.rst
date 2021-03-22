Custom kernels
==============

.. autosummary::
   :toctree: generated/

   cupy.ElementwiseKernel
   cupy.ReductionKernel
   cupy.RawKernel
   cupy.RawModule
   cupy.fuse


JIT kernel definition
---------------------

.. autosummary::
   :toctree: generated/

   cupyx.jit.rawkernel
   cupyx.jit.threadIdx
   cupyx.jit.blockDim
   cupyx.jit.blockIdx
   cupyx.jit.gridDim
   cupyx.jit.syncthreads
   cupyx.jit.shared_memory
   cupyx.jit._interface._JitRawKernel


Kernel binary memoization
-------------------------

.. autosummary::
   :toctree: generated/

   cupy.memoize
   cupy.clear_memo
