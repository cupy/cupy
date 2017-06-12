Multi-Dimensional Array (ndarray)
=================================

:class:`cupy.ndarray` is the CuPy counterpart of NumPy :class:`numpy.ndarray`.
It provides an intuitive interface for a fixed-size multidimensional array which resides
in a CUDA device.

For the basic concept of ``ndarray``\s, please refer to the `NumPy documentation <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.ndarray


Code compatibility features
---------------------------

:class:`cupy.ndarray` is designed to be interchangeable with :class:`numpy.ndarray` in terms of code compatibility as much as possible.
But occasionally, you will need to know whether the arrays you're handling are :class:`cupy.ndarray` or :class:`numpy.ndarray`.
One example is when invoking module-level functions such as :meth:`cupy.sum` or :meth:`numpy.sum`.
In such situations, :meth:`cupy.get_array_module` can be used.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.get_array_module


Conversion to/from NumPy arrays
-------------------------------

:class:`cupy.ndarray` and :class:`numpy.ndarray` are not implicitly convertible to each other.
That means, NumPy functions cannot take :class:`cupy.ndarray`\s as inputs, and vice versa.

- To convert :class:`numpy.ndarray` to :class:`cupy.ndarray`, use :meth:`cupy.array` or :meth:`cupy.asarray`.
- To convert :class:`cupy.ndarray` to :class:`numpy.ndarray`, use :meth:`cupy.asnumpy` or :meth:`cupy.ndarray.get`.

Note that converting between :class:`cupy.ndarray` and :class:`numpy.ndarray` incurs data transfer between
the host (CPU) device and the GPU device, which is costly in terms of performance.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.array
   cupy.asarray
   cupy.asnumpy
