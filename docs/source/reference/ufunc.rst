Universal functions (:class:`cupy.ufunc`)
=========================================

.. Hint:: `NumPy API Reference: Universal functions (numpy.ufunc) <https://numpy.org/doc/stable/reference/ufuncs.html>`_

.. currentmodule:: cupy

CuPy provides universal functions (a.k.a. ufuncs) to support various elementwise operations.
CuPy's ufunc supports following features of NumPy's one:

- Broadcasting
- Output type determination
- Casting rules

CuPy's ufunc currently does not provide methods such as ``reduce``, ``accumulate``, ``reduceat``, ``outer``, and ``at``.


ufunc
-----

.. autosummary::
   :toctree: generated/

   ufunc


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
   power
   remainder
   mod
   fmod
   absolute
   rint
   sign
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


ufunc.at
--------

Currently, CuPy does not support ``at`` for ufuncs in general.
However, :func:`cupyx.scatter_add` can substitute ``add.at`` as both behave identically.
