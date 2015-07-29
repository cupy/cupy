Utilities
=========

CUDA utilities
--------------
.. automodule:: chainer.cuda

Initialization and global states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: init

Devices and contexts
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: get_device
.. autofunction:: use_device
.. autofunction:: using_device
.. autoclass:: DeviceUser

GPUArray allocation and copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: copy
.. autofunction:: copy_async

.. autofunction:: empty
.. autofunction:: empty_like
.. autofunction:: full
.. autofunction:: full_like
.. autofunction:: zeros
.. autofunction:: zeros_like
.. autofunction:: ones
.. autofunction:: ones_like

.. autofunction:: to_cpu
.. autofunction:: to_cpu_async
.. autofunction:: to_gpu
.. autofunction:: to_gpu_async

Kernel definition utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: elementwise
.. autofunction:: reduce

Common algorithms
-----------------
.. automodule:: chainer.utils

.. autoclass:: WalkerAlias
   :members: sample, to_gpu
