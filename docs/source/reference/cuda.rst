Low-level CUDA support
======================

.. _device_management:

Device management
-----------------

.. autosummary::
   :toctree: generated/

   cupy.cuda.Device


Memory management
-----------------

.. autosummary::
   :toctree: generated/

   cupy.get_default_memory_pool
   cupy.get_default_pinned_memory_pool
   cupy.cuda.Memory
   cupy.cuda.MemoryAsync
   cupy.cuda.ManagedMemory
   cupy.cuda.UnownedMemory
   cupy.cuda.PinnedMemory
   cupy.cuda.MemoryPointer
   cupy.cuda.PinnedMemoryPointer
   cupy.cuda.malloc_managed
   cupy.cuda.malloc_async
   cupy.cuda.alloc
   cupy.cuda.alloc_pinned_memory
   cupy.cuda.get_allocator
   cupy.cuda.set_allocator
   cupy.cuda.using_allocator
   cupy.cuda.set_pinned_memory_allocator
   cupy.cuda.MemoryPool
   cupy.cuda.MemoryAsyncPool
   cupy.cuda.PinnedMemoryPool
   cupy.cuda.PythonFunctionAllocator
   cupy.cuda.CFunctionAllocator


Memory hook
-----------

.. autosummary::
   :toctree: generated/

   cupy.cuda.MemoryHook
   cupy.cuda.memory_hooks.DebugPrintHook
   cupy.cuda.memory_hooks.LineProfileHook


.. _stream_event_api:

Streams and events
------------------

.. autosummary::
   :toctree: generated/

   cupy.cuda.Stream
   cupy.cuda.ExternalStream
   cupy.cuda.get_current_stream
   cupy.cuda.Event
   cupy.cuda.get_elapsed_time


.. _graph_api:

Graphs
------

.. autosummary::
   :toctree: generated/

   cupy.cuda.Graph


Texture and surface memory
--------------------------

.. autosummary::
   :toctree: generated/

   cupy.cuda.texture.ChannelFormatDescriptor
   cupy.cuda.texture.CUDAarray
   cupy.cuda.texture.ResourceDescriptor
   cupy.cuda.texture.TextureDescriptor
   cupy.cuda.texture.TextureObject
   cupy.cuda.texture.SurfaceObject


NVTX
----

.. autosummary::
   :toctree: generated/

   cupy.cuda.nvtx.Mark
   cupy.cuda.nvtx.MarkC
   cupy.cuda.nvtx.RangePush
   cupy.cuda.nvtx.RangePushC
   cupy.cuda.nvtx.RangePop


NCCL
----

.. autosummary::
   :toctree: generated/

   cupy.cuda.nccl.NcclCommunicator
   cupy.cuda.nccl.get_build_version
   cupy.cuda.nccl.get_version
   cupy.cuda.nccl.get_unique_id
   cupy.cuda.nccl.groupStart
   cupy.cuda.nccl.groupEnd


Version
-------

.. autosummary::
   :toctree: generated/

   cupy.cuda.get_local_runtime_version

.. _runtime_api:

Runtime API
-----------

CuPy wraps CUDA Runtime APIs to provide the native CUDA operations.
Please check the `CUDA Runtime API documentation <https://docs.nvidia.com/cuda/cuda-runtime-api/index.html>`_
to use these functions.

.. autosummary::
   :toctree: generated/

   cupy.cuda.runtime.driverGetVersion
   cupy.cuda.runtime.runtimeGetVersion
   cupy.cuda.runtime.getDevice
   cupy.cuda.runtime.getDeviceProperties
   cupy.cuda.runtime.deviceGetAttribute
   cupy.cuda.runtime.deviceGetByPCIBusId
   cupy.cuda.runtime.deviceGetPCIBusId
   cupy.cuda.runtime.deviceGetDefaultMemPool
   cupy.cuda.runtime.deviceGetMemPool
   cupy.cuda.runtime.deviceSetMemPool
   cupy.cuda.runtime.memPoolCreate
   cupy.cuda.runtime.memPoolDestroy
   cupy.cuda.runtime.memPoolTrimTo
   cupy.cuda.runtime.getDeviceCount
   cupy.cuda.runtime.setDevice
   cupy.cuda.runtime.deviceSynchronize
   cupy.cuda.runtime.deviceCanAccessPeer
   cupy.cuda.runtime.deviceEnablePeerAccess
   cupy.cuda.runtime.deviceGetLimit
   cupy.cuda.runtime.deviceSetLimit
   cupy.cuda.runtime.malloc
   cupy.cuda.runtime.mallocManaged
   cupy.cuda.runtime.malloc3DArray
   cupy.cuda.runtime.mallocArray
   cupy.cuda.runtime.mallocAsync
   cupy.cuda.runtime.mallocFromPoolAsync
   cupy.cuda.runtime.hostAlloc
   cupy.cuda.runtime.hostRegister
   cupy.cuda.runtime.hostUnregister
   cupy.cuda.runtime.free
   cupy.cuda.runtime.freeHost
   cupy.cuda.runtime.freeArray
   cupy.cuda.runtime.freeAsync
   cupy.cuda.runtime.memGetInfo
   cupy.cuda.runtime.memcpy
   cupy.cuda.runtime.memcpyAsync
   cupy.cuda.runtime.memcpyPeer
   cupy.cuda.runtime.memcpyPeerAsync
   cupy.cuda.runtime.memcpy2D
   cupy.cuda.runtime.memcpy2DAsync
   cupy.cuda.runtime.memcpy2DFromArray
   cupy.cuda.runtime.memcpy2DFromArrayAsync
   cupy.cuda.runtime.memcpy2DToArray
   cupy.cuda.runtime.memcpy2DToArrayAsync
   cupy.cuda.runtime.memcpy3D
   cupy.cuda.runtime.memcpy3DAsync
   cupy.cuda.runtime.memset
   cupy.cuda.runtime.memsetAsync
   cupy.cuda.runtime.memPrefetchAsync
   cupy.cuda.runtime.memAdvise
   cupy.cuda.runtime.pointerGetAttributes
   cupy.cuda.runtime.streamCreate
   cupy.cuda.runtime.streamCreateWithFlags
   cupy.cuda.runtime.streamDestroy
   cupy.cuda.runtime.streamSynchronize
   cupy.cuda.runtime.streamAddCallback
   cupy.cuda.runtime.streamQuery
   cupy.cuda.runtime.streamWaitEvent
   cupy.cuda.runtime.launchHostFunc
   cupy.cuda.runtime.eventCreate
   cupy.cuda.runtime.eventCreateWithFlags
   cupy.cuda.runtime.eventDestroy
   cupy.cuda.runtime.eventElapsedTime
   cupy.cuda.runtime.eventQuery
   cupy.cuda.runtime.eventRecord
   cupy.cuda.runtime.eventSynchronize
   cupy.cuda.runtime.ipcGetMemHandle
   cupy.cuda.runtime.ipcOpenMemHandle
   cupy.cuda.runtime.ipcCloseMemHandle
   cupy.cuda.runtime.ipcGetEventHandle
   cupy.cuda.runtime.ipcOpenEventHandle
   cupy.cuda.runtime.profilerStart
   cupy.cuda.runtime.profilerStop
