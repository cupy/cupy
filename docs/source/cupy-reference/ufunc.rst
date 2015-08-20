Universal Functions (ufunc)
===========================

CuPy provides universal functions (a.k.a. ufuncs) to support various elementwise operations.
CuPy's ufunc supports following features of NumPy's one:

- Broadcasting
- Output type determination
- Casting rules

CuPy's ufunc currently does not provide methods such as ``reduce``, ``accumulate``, ``reduceat``, ``outer``, and ``at``.


Ufunc class
-----------

.. autoclass:: cupy.ufunc
   :members:


Available ufuncs
----------------

Math operations
~~~~~~~~~~~~~~~

.. module:: cupy

:data:`add`
:data:`subtract`
:data:`multiply`
:data:`divide`
:data:`logaddexp`
:data:`logaddexp2`
:data:`true_divide`
:data:`floor_divide`
:data:`negative`
:data:`power`
:data:`remainder`
:data:`mod`
:data:`fmod`
:data:`absolute`
:data:`rint`
:data:`sign`
:data:`exp`
:data:`exp2`
:data:`log`
:data:`log2`
:data:`log10`
:data:`expm1`
:data:`log1p`
:data:`sqrt`
:data:`square`
:data:`reciprocal`

Trigonometric functions
~~~~~~~~~~~~~~~~~~~~~~~

:data:`sin`
:data:`cos`
:data:`tan`
:data:`arcsin`
:data:`arccos`
:data:`arctan`
:data:`arctan2`
:data:`hypot`
:data:`sinh`
:data:`cosh`
:data:`tanh`
:data:`arcsinh`
:data:`arccosh`
:data:`arctanh`
:data:`deg2rad`
:data:`rad2deg`

Bit-twiddling functions
~~~~~~~~~~~~~~~~~~~~~~~

:data:`bitwise_and`
:data:`bitwise_or`
:data:`bitwise_xor`
:data:`invert`
:data:`left_shift`
:data:`right_shift`

Comparison functions
~~~~~~~~~~~~~~~~~~~~

:data:`greater`
:data:`greater_equal`
:data:`less`
:data:`less_equal`
:data:`not_equal`
:data:`equal`
:data:`logical_and`
:data:`logical_or`
:data:`logical_xor`
:data:`logical_not`
:data:`maximum`
:data:`minimum`
:data:`fmax`
:data:`fmin`

Floating point values
~~~~~~~~~~~~~~~~~~~~~

:data:`isfinite`
:data:`isinf`
:data:`isnan`
:data:`signbit`
:data:`copysign`
:data:`nextafter`
:data:`modf`
:data:`ldexp`
:data:`frexp`
:data:`fmod`
:data:`floor`
:data:`ceil`
:data:`trunc`
