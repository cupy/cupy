Utilities
=========

CUDA utilities
--------------
.. automodule:: chainer.cuda

Devices
~~~~~~~
.. autofunction:: get_device

CuPy array allocation and copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   As of v1.3.0, the following array construction wrappers are marked as
   deprecated. Use the corresponding functions of the :mod:`cupy` module
   instead. The main difference of them is that the default dtype is changed
   from float32 to float64.

   ============================= =========================
    Deprecated functions          Recommended functions
   ============================= =========================
    ``chainer.cuda.empty``        :func:`cupy.empty`
    ``chainer.cuda.empty_like``   :func:`cupy.empty_like`
    ``chainer.cuda.zeros``        :func:`cupy.zeros`
    ``chainer.cuda.zeros_like``   :func:`cupy.zeros_like`
    ``chainer.cuda.ones``         :func:`cupy.ones`
    ``chainer.cuda.ones_like``    :func:`cupy.ones_like`
    ``chainer.cuda.full``         :func:`cupy.full`
    ``chainer.cuda.full_like``    :func:`cupy.full_like`
   ============================= =========================

.. autofunction:: copy
.. autofunction:: to_cpu
.. autofunction:: to_gpu

Kernel definition utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: memoize
.. autofunction:: elementwise
.. autofunction:: reduce

CPU/GPU generic code support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: get_array_module

cuDNN support
~~~~~~~~~~~~~
.. autofunction:: set_max_workspace_size
.. autofunction:: get_max_workspace_size

Common algorithms
-----------------
.. automodule:: chainer.utils

.. autoclass:: WalkerAlias
   :members: sample, to_gpu
