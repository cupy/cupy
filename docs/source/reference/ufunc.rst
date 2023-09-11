Universal functions (:class:`cupy.ufunc`)
=========================================

.. Hint:: `NumPy API Reference: Universal functions (numpy.ufunc) <https://numpy.org/doc/stable/reference/ufuncs.html>`_

.. currentmodule:: cupy

CuPy provides universal functions (a.k.a. ufuncs) to support various elementwise operations.
CuPy's ufunc supports following features of NumPy's one:

- Broadcasting
- Output type determination
- Casting rules


ufunc
-----

.. autosummary::
   :toctree: generated/

   ufunc

Methods
~~~~~~~

These methods are only available for selected ufuncs.

* :meth:`ufunc.reduce <cupy.ufunc.reduce>`: :func:`~cupy.add`, :func:`~cupy.multiply`
* :meth:`ufunc.accumulate <cupy.ufunc.accumulate>`: :func:`~cupy.add`, :func:`~cupy.multiply`
* :meth:`ufunc.reduceat <ufunc.reduceat>`: :func:`~cupy.add`
* :meth:`ufunc.outer <cupy.ufunc.outer>`: All ufuncs
* :meth:`ufunc.at <cupy.ufunc.at>`: :func:`~cupy.add`, :func:`~cupy.subtract`, :func:`~cupy.maximum`, :func:`~cupy.minimum`, :func:`~cupy.bitwise_and`, :func:`~cupy.bitwise_or`, :func:`~cupy.bitwise_xor`

.. hint::

   In case you need support for other ufuncs, submit a feature request along with your use-case in `the tracker issue <https://github.com/cupy/cupy/issues/7082>`_.


Available ufuncs
----------------

Math operations
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   add
   subtract
   multiply
   matmul
   divide
   logaddexp
   logaddexp2
   true_divide
   floor_divide
   negative
   positive
   power
   float_power
   remainder
   mod
   fmod
   divmod
   absolute
   fabs
   rint
   sign
   heaviside
   conj
   conjugate
   exp
   exp2
   log
   log2
   log10
   expm1
   log1p
   sqrt
   square
   cbrt
   reciprocal
   gcd
   lcm


Trigonometric functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   sin
   cos
   tan
   arcsin
   arccos
   arctan
   arctan2
   hypot
   sinh
   cosh
   tanh
   arcsinh
   arccosh
   arctanh
   degrees
   radians
   deg2rad
   rad2deg


Bit-twiddling functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   bitwise_and
   bitwise_or
   bitwise_xor
   invert
   left_shift
   right_shift


Comparison functions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   greater
   greater_equal
   less
   less_equal
   not_equal
   equal
   logical_and
   logical_or
   logical_xor
   logical_not
   maximum
   minimum
   fmax
   fmin


Floating functions
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   isfinite
   isinf
   isnan
   fabs
   signbit
   copysign
   nextafter
   modf
   ldexp
   frexp
   fmod
   floor
   ceil
   trunc


Generalized Universal Functions
-------------------------------

.. currentmodule:: cupyx

In addition to regular ufuncs, CuPy also provides a wrapper class to convert
regular cupy functions into Generalized Universal Functions as in NumPy `<https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html>`_.
This allows to automatically use keyword arguments such as ``axes``, ``order``, ``dtype``
without needing to explicitly implement them in the wrapped function.


.. autosummary::
   :toctree: generated/

   GeneralizedUFunc
