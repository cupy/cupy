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

   cupy.get_default_memory_pool
   cupy.get_default_pinned_memory_pool
   cupy.cuda.Memory
   cupy.cuda.PinnedMemory
   cupy.cuda.MemoryPointer
   cupy.cuda.PinnedMemoryPointer
   cupy.cuda.alloc
   cupy.cuda.alloc_pinned_memory
   cupy.cuda.set_allocator
   cupy.cuda.set_pinned_memory_allocator
   cupy.cuda.MemoryPool
   cupy.cuda.PinnedMemoryPool
   cupy.cuda.SingleDeviceMemoryPool


Memory hook
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.cuda.MemoryHook
   cupy.cuda.memory_hooks.DebugPrintHook
   cupy.cuda.memory_hooks.LineProfileHook

Streams and events
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.cuda.Stream
   cupy.cuda.get_current_stream
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
