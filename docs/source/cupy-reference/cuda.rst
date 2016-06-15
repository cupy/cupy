Low-Level CUDA Support
======================

Device management
-----------------

.. autoclass:: cupy.cuda.Device
   :members:


Memory management
-----------------

.. autoclass:: cupy.cuda.Memory
   :members:
.. autoclass:: cupy.cuda.MemoryPointer
   :members:
.. autofunction:: cupy.cuda.alloc
.. autofunction:: cupy.cuda.set_allocator
.. autoclass:: cupy.cuda.MemoryPool
   :members:


Streams and events
------------------

.. autoclass:: cupy.cuda.Stream
   :members:
.. autoclass:: cupy.cuda.Event
   :members:

.. autofunction:: cupy.cuda.get_elapsed_time


Profiler
--------

.. autofunction:: cupy.cuda.profile
.. autofunction:: cupy.cuda.profiler.initialize
.. autofunction:: cupy.cuda.profiler.start
.. autofunction:: cupy.cuda.profiler.stop
