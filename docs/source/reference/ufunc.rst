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

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.ufunc


Available ufuncs
----------------

Math operations
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.add
   cupy.subtract
   cupy.multiply
   cupy.divide
   cupy.logaddexp
   cupy.logaddexp2
   cupy.true_divide
   cupy.floor_divide
   cupy.negative
   cupy.power
   cupy.remainder
   cupy.mod
   cupy.fmod
   cupy.absolute
   cupy.rint
   cupy.sign
   cupy.exp
   cupy.exp2
   cupy.log
   cupy.log2
   cupy.log10
   cupy.expm1
   cupy.log1p
   cupy.sqrt
   cupy.square
   cupy.reciprocal


Trigonometric functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.sin
   cupy.cos
   cupy.tan
   cupy.arcsin
   cupy.arccos
   cupy.arctan
   cupy.arctan2
   cupy.hypot
   cupy.sinh
   cupy.cosh
   cupy.tanh
   cupy.arcsinh
   cupy.arccosh
   cupy.arctanh
   cupy.deg2rad
   cupy.rad2deg


Bit-twiddling functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.bitwise_and
   cupy.bitwise_or
   cupy.bitwise_xor
   cupy.invert
   cupy.left_shift
   cupy.right_shift


Comparison functions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.greater
   cupy.greater_equal
   cupy.less
   cupy.less_equal
   cupy.not_equal
   cupy.equal
   cupy.logical_and
   cupy.logical_or
   cupy.logical_xor
   cupy.logical_not
   cupy.maximum
   cupy.minimum
   cupy.fmax
   cupy.fmin


Floating point values
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.isfinite
   cupy.isinf
   cupy.isnan
   cupy.signbit
   cupy.copysign
   cupy.nextafter
   cupy.modf
   cupy.ldexp
   cupy.frexp
   cupy.fmod
   cupy.floor
   cupy.ceil
   cupy.trunc


ufunc.at
--------

Currently, CuPy does not support ``at`` for ufuncs in general.
However, :func:`cupy.scatter_add` can substitute ``add.at`` as both behave identically.
