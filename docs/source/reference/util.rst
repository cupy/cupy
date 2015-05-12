Utilities
=========

CUDA utilities
--------------
.. automodule:: chainer.cuda

Initialization and global states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: init
.. autofunction:: shutdown
.. autofunction:: mem_alloc

Devices and contexts
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: get_device
.. autofunction:: use_device
.. autofunction:: using_device
.. autoclass:: DeviceUser

.. autofunction:: get_context

.. autofunction:: get_cublas_handle
.. autofunction:: using_cumisc
.. autoclass:: CumiscUser

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

Random number generators
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: get_generator
.. autofunction:: seed

Kernel definition utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: elementwise
.. autofunction:: reduce

Interprocess communication on GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: IPCEvent
.. autoclass:: IPCArrayHandle

CuDNN utilities
---------------
.. automodule:: chainer.cudnn

.. autodata:: available
.. autodata:: enabled

.. autofunction:: get_default_handle

.. autofunction:: get_ptr
.. autofunction:: get_conv_bias_desc
.. autofunction:: get_conv2d_desc
.. autofunction:: get_filter4d_desc
.. autofunction:: get_pool2d_desc
.. autofunction:: get_tensor_desc

.. autofunction:: shutdown

Gradient checking utilities
---------------------------
.. automodule:: chainer.gradient_check

.. autofunction:: assert_allclose
.. autofunction:: numerical_grad
