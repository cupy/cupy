Low-Level CUDA Support
======================

Device management
-----------------

.. autoclass:: cupy.cuda.Device
   :members:
.. autofunction:: cupy.cuda.clear_device_dependent_memo
.. autofunction:: cupy.cuda.memoize
.. autofunction:: cupy.cuda.using_device


Memory management
-----------------

.. autoclass:: cupy.cuda.Memory
   :members:
.. autoclass:: cupy.cuda.MemoryPointer
   :members:
.. autofunction:: cupy.cuda.alloc
.. autofunction:: cupy.cuda.set_default_allocator
.. autoclass:: cupy.cuda.MemoryPool
   :members:


Streams and events
------------------

.. autoclass:: cupy.cuda.Stream
   :members:
.. autoclass:: cupy.cuda.Event
   :members:

.. autofunction:: cupy.cuda.get_elapsed_time
