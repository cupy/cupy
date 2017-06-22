Low-Level CUDA Support
======================

Device management
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.cuda.Device


Memory management
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.cuda.Memory
   cupy.cuda.MemoryPointer
   cupy.cuda.alloc
   cupy.cuda.set_allocator
   cupy.cuda.MemoryPool


Streams and events
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.cuda.Stream
   cupy.cuda.Event
   cupy.cuda.get_elapsed_time


Profiler
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.cuda.profile
   cupy.cuda.profiler.initialize
   cupy.cuda.profiler.start
   cupy.cuda.profiler.stop
   cupy.cuda.nvtx.Mark
   cupy.cuda.nvtx.MarkC
   cupy.cuda.nvtx.RangePush
   cupy.cuda.nvtx.RangePushC
   cupy.cuda.nvtx.RangePop
