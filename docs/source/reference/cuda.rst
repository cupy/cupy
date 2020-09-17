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
   cupy.cuda.UnownedMemory
   cupy.cuda.PinnedMemory
   cupy.cuda.MemoryPointer
   cupy.cuda.PinnedMemoryPointer
   cupy.cuda.alloc
   cupy.cuda.alloc_pinned_memory
   cupy.cuda.get_allocator
   cupy.cuda.set_allocator
   cupy.cuda.using_allocator
   cupy.cuda.set_pinned_memory_allocator
   cupy.cuda.MemoryPool
   cupy.cuda.PinnedMemoryPool
   cupy.cuda.PythonFunctionAllocator


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
   cupy.cuda.ExternalStream
   cupy.cuda.get_current_stream
   cupy.cuda.Event
   cupy.cuda.get_elapsed_time


Texture and surface memory
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.cuda.texture.ChannelFormatDescriptor
   cupy.cuda.texture.CUDAarray
   cupy.cuda.texture.ResourceDescriptor
   cupy.cuda.texture.TextureDescriptor
   cupy.cuda.texture.TextureObject
   cupy.cuda.texture.SurfaceObject
   cupy.cuda.texture.TextureReference


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


NCCL
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.cuda.nccl.NcclCommunicator
   cupy.cuda.nccl.get_build_version
   cupy.cuda.nccl.get_version
   cupy.cuda.nccl.get_unique_id
   cupy.cuda.nccl.groupStart
   cupy.cuda.nccl.groupEnd


Runtime API
-----------

CuPy wraps CUDA Runtime APIs to provide the native CUDA operations.
Please check the `Original CUDA Runtime API document <https://docs.nvidia.com/cuda/cuda-runtime-api/index.html>`_
to use these functions.



.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy_backends.cuda.api.runtime.driverGetVersion
   cupy_backends.cuda.api.runtime.runtimeGetVersion
   cupy_backends.cuda.api.runtime.getDevice
   cupy_backends.cuda.api.runtime.deviceGetAttribute
   cupy_backends.cuda.api.runtime.deviceGetByPCIBusId
   cupy_backends.cuda.api.runtime.deviceGetPCIBusId
   cupy_backends.cuda.api.runtime.getDeviceCount
   cupy_backends.cuda.api.runtime.setDevice
   cupy_backends.cuda.api.runtime.deviceSynchronize
   cupy_backends.cuda.api.runtime.deviceCanAccessPeer
   cupy_backends.cuda.api.runtime.deviceEnablePeerAccess
   cupy_backends.cuda.api.runtime.deviceGetLimit
   cupy_backends.cuda.api.runtime.deviceSetLimit
   cupy_backends.cuda.api.runtime.malloc
   cupy_backends.cuda.api.runtime.mallocManaged
   cupy_backends.cuda.api.runtime.malloc3DArray
   cupy_backends.cuda.api.runtime.mallocArray
   cupy_backends.cuda.api.runtime.hostAlloc
   cupy_backends.cuda.api.runtime.hostRegister
   cupy_backends.cuda.api.runtime.hostUnregister
   cupy_backends.cuda.api.runtime.free
   cupy_backends.cuda.api.runtime.freeHost
   cupy_backends.cuda.api.runtime.freeArray
   cupy_backends.cuda.api.runtime.memGetInfo
   cupy_backends.cuda.api.runtime.memcpy
   cupy_backends.cuda.api.runtime.memcpyAsync
   cupy_backends.cuda.api.runtime.memcpyPeer
   cupy_backends.cuda.api.runtime.memcpyPeerAsync
   cupy_backends.cuda.api.runtime.memcpy2D
   cupy_backends.cuda.api.runtime.memcpy2DAsync
   cupy_backends.cuda.api.runtime.memcpy2DFromArray
   cupy_backends.cuda.api.runtime.memcpy2DFromArrayAsync
   cupy_backends.cuda.api.runtime.memcpy2DToArray
   cupy_backends.cuda.api.runtime.memcpy2DToArrayAsync
   cupy_backends.cuda.api.runtime.memcpy3D
   cupy_backends.cuda.api.runtime.memcpy3DAsync
   cupy_backends.cuda.api.runtime.memset
   cupy_backends.cuda.api.runtime.memsetAsync
   cupy_backends.cuda.api.runtime.memPrefetchAsync
   cupy_backends.cuda.api.runtime.memAdvise
   cupy_backends.cuda.api.runtime.pointerGetAttributes
   cupy_backends.cuda.api.runtime.streamCreate
   cupy_backends.cuda.api.runtime.streamCreateWithFlags
   cupy_backends.cuda.api.runtime.streamDestroy
   cupy_backends.cuda.api.runtime.streamSynchronize
   cupy_backends.cuda.api.runtime.streamAddCallback
   cupy_backends.cuda.api.runtime.streamQuery
   cupy_backends.cuda.api.runtime.streamWaitEvent
   cupy_backends.cuda.api.runtime.eventCreate
   cupy_backends.cuda.api.runtime.eventCreateWithFlags
   cupy_backends.cuda.api.runtime.eventDestroy
   cupy_backends.cuda.api.runtime.eventElapsedTime
   cupy_backends.cuda.api.runtime.eventQuery
   cupy_backends.cuda.api.runtime.eventRecord
   cupy_backends.cuda.api.runtime.eventSynchronize
   cupy_backends.cuda.api.runtime.ipcGetMemHandle
   cupy_backends.cuda.api.runtime.ipcOpenMemHandle
   cupy_backends.cuda.api.runtime.ipcCloseMemHandle
   cupy_backends.cuda.api.runtime.ipcGetEventHandle
   cupy_backends.cuda.api.runtime.ipcOpenEventHandle
