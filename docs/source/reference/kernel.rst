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

Supported Python built-in functions include: :obj:`range`, :func:`len`, :func:`max`, :func:`min`.

.. note:: If loop unrolling is needed, use :func:`cupyx.jit.range` instead of the built-in :obj:`range`.

.. autosummary::
   :toctree: generated/

   cupyx.jit.rawkernel
   cupyx.jit.threadIdx
   cupyx.jit.blockDim
   cupyx.jit.blockIdx
   cupyx.jit.gridDim
   cupyx.jit.grid
   cupyx.jit.gridsize
   cupyx.jit.laneid
   cupyx.jit.warpsize
   cupyx.jit.range
   cupyx.jit.syncthreads
   cupyx.jit.syncwarp
   cupyx.jit.shfl_sync
   cupyx.jit.shfl_up_sync
   cupyx.jit.shfl_down_sync
   cupyx.jit.shfl_xor_sync
   cupyx.jit.shared_memory
   cupyx.jit.atomic_add
   cupyx.jit.atomic_sub
   cupyx.jit.atomic_exch
   cupyx.jit.atomic_min
   cupyx.jit.atomic_max
   cupyx.jit.atomic_inc
   cupyx.jit.atomic_dec
   cupyx.jit.atomic_cas
   cupyx.jit.atomic_and
   cupyx.jit.atomic_or
   cupyx.jit.atomic_xor
   cupyx.jit.cg.this_grid
   cupyx.jit.cg.this_thread_block
   cupyx.jit.cg.sync
   cupyx.jit.cg.memcpy_async
   cupyx.jit.cg.wait
   cupyx.jit.cg.wait_prior
   cupyx.jit._interface._JitRawKernel


Kernel binary memoization
-------------------------

.. autosummary::
   :toctree: generated/

   cupy.memoize
   cupy.clear_memo
