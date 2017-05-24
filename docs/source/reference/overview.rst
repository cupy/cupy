Support features
================

A list of supported attributes, properties, and methods of ndarray
------------------------------------------------------------------

Memory layout
~~~~~~~~~~~~~

:attr:`~ndarray.base`
:attr:`~ndarray.ctypes`
:attr:`~ndarray.itemsize`
:attr:`~ndarray.flags`
:attr:`~ndarray.nbytes`
:attr:`~ndarray.shape`
:attr:`~ndarray.size`
:attr:`~ndarray.strides`

Data type
~~~~~~~~~

:attr:`~ndarray.dtype`

Other attributes
~~~~~~~~~~~~~~~~

:attr:`~ndarray.T`

Array conversion
~~~~~~~~~~~~~~~~

:meth:`~ndarray.tolist`
:meth:`~ndarray.tofile`
:meth:`~ndarray.dump`
:meth:`~ndarray.dumps`
:meth:`~ndarray.astype`
:meth:`~ndarray.copy`
:meth:`~ndarray.view`
:meth:`~ndarray.fill`

Shape manipulation
~~~~~~~~~~~~~~~~~~

:meth:`~ndarray.reshape`
:meth:`~ndarray.transpose`
:meth:`~ndarray.swapaxes`
:meth:`~ndarray.ravel`
:meth:`~ndarray.squeeze`

Item selection and manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~ndarray.take`
:meth:`~ndarray.diagonal`

Calculation
~~~~~~~~~~~

:meth:`~ndarray.max`
:meth:`~ndarray.argmax`
:meth:`~ndarray.min`
:meth:`~ndarray.argmin`
:meth:`~ndarray.clip`
:meth:`~ndarray.trace`
:meth:`~ndarray.sum`
:meth:`~ndarray.mean`
:meth:`~ndarray.var`
:meth:`~ndarray.std`
:meth:`~ndarray.prod`
:meth:`~ndarray.dot`

Arithmetic and comparison operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~ndarray.__lt__`
:meth:`~ndarray.__le__`
:meth:`~ndarray.__gt__`
:meth:`~ndarray.__ge__`
:meth:`~ndarray.__eq__`
:meth:`~ndarray.__ne__`
:meth:`~ndarray.__nonzero__`
:meth:`~ndarray.__neg__`
:meth:`~ndarray.__pos__`
:meth:`~ndarray.__abs__`
:meth:`~ndarray.__invert__`
:meth:`~ndarray.__add__`
:meth:`~ndarray.__sub__`
:meth:`~ndarray.__mul__`
:meth:`~ndarray.__div__`
:meth:`~ndarray.__truediv__`
:meth:`~ndarray.__floordiv__`
:meth:`~ndarray.__mod__`
:meth:`~ndarray.__divmod__`
:meth:`~ndarray.__pow__`
:meth:`~ndarray.__lshift__`
:meth:`~ndarray.__rshift__`
:meth:`~ndarray.__and__`
:meth:`~ndarray.__or__`
:meth:`~ndarray.__xor__`
:meth:`~ndarray.__iadd__`
:meth:`~ndarray.__isub__`
:meth:`~ndarray.__imul__`
:meth:`~ndarray.__idiv__`
:meth:`~ndarray.__itruediv__`
:meth:`~ndarray.__ifloordiv__`
:meth:`~ndarray.__imod__`
:meth:`~ndarray.__ipow__`
:meth:`~ndarray.__ilshift__`
:meth:`~ndarray.__irshift__`
:meth:`~ndarray.__iand__`
:meth:`~ndarray.__ior__`
:meth:`~ndarray.__ixor__`

Special methods
~~~~~~~~~~~~~~~

:meth:`~ndarray.__copy__`
:meth:`~ndarray.__deepcopy__`
:meth:`~ndarray.__reduce__`
:meth:`~ndarray.__array__`
:meth:`~ndarray.__len__`
:meth:`~ndarray.__getitem__`
:meth:`~ndarray.__setitem__`
:meth:`~ndarray.__int__`
:meth:`~ndarray.__long__`
:meth:`~ndarray.__float__`
:meth:`~ndarray.__oct__`
:meth:`~ndarray.__hex__`
:meth:`~ndarray.__repr__`
:meth:`~ndarray.__str__`

Memory transfer
~~~~~~~~~~~~~~~

:meth:`~ndarray.get`
:meth:`~ndarray.set`


A list of supported routines of :mod:`cupy` module
--------------------------------------------------

Array creation routines
~~~~~~~~~~~~~~~~~~~~~~~

:func:`empty`
:func:`empty_like`
:func:`eye`
:func:`identity`
:func:`ones`
:func:`ones_like`
:func:`zeros`
:func:`zeros_like`
:func:`full`
:func:`full_like`

:func:`array`
:func:`asarray`
:func:`ascontiguousarray`
:func:`copy`

:func:`arange`
:func:`linspace`

:func:`diag`
:func:`diagflat`

Array manipulation routines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`copyto`

:func:`reshape`
:func:`ravel`

:func:`rollaxis`
:func:`swapaxes`
:func:`transpose`

:func:`atleast_1d`
:func:`atleast_2d`
:func:`atleast_3d`
:class:`broadcast`
:func:`broadcast_arrays`
:func:`broadcast_to`
:func:`expand_dims`
:func:`squeeze`

:func:`column_stack`
:func:`concatenate`
:func:`dstack`
:func:`hstack`
:func:`vstack`

:func:`array_split`
:func:`dsplit`
:func:`hsplit`
:func:`split`
:func:`vsplit`

:func:`roll`

Binary operations
~~~~~~~~~~~~~~~~~

:data:`bitwise_and`
:data:`bitwise_or`
:data:`bitwise_xor`
:data:`invert`
:data:`left_shift`
:data:`right_shift`

Indexing routines
~~~~~~~~~~~~~~~~~

:func:`take`
:func:`diagonal`

Input and output
~~~~~~~~~~~~~~~~

:func:`load`
:func:`save`
:func:`savez`
:func:`savez_compressed`

:func:`array_repr`
:func:`array_str`

Linear algebra
~~~~~~~~~~~~~~

:func:`dot`
:func:`vdot`
:func:`inner`
:func:`outer`
:func:`tensordot`

:func:`trace`

Logic functions
~~~~~~~~~~~~~~~

:data:`isfinite`
:data:`isinf`
:data:`isnan`

:data:`logical_and`
:data:`logical_or`
:data:`logical_not`
:data:`logical_xor`

:data:`greater`
:data:`greater_equal`
:data:`less`
:data:`less_equal`
:data:`equal`
:data:`not_equal`

Mathematical functions
~~~~~~~~~~~~~~~~~~~~~~

:data:`sin`
:data:`cos`
:data:`tan`
:data:`arcsin`
:data:`arccos`
:data:`arctan`
:data:`hypot`
:data:`arctan2`
:data:`deg2rad`
:data:`rad2deg`
:data:`degrees`
:data:`radians`

:data:`sinh`
:data:`cosh`
:data:`tanh`
:data:`arcsinh`
:data:`arccosh`
:data:`arctanh`

:data:`rint`
:data:`floor`
:data:`ceil`
:data:`trunc`

:func:`sum`
:func:`prod`

:data:`exp`
:data:`expm1`
:data:`exp2`
:data:`log`
:data:`log10`
:data:`log2`
:data:`log1p`
:data:`logaddexp`
:data:`logaddexp2`

:data:`signbit`
:data:`copysign`
:data:`ldexp`
:data:`frexp`
:data:`nextafter`

:data:`add`
:data:`reciprocal`
:data:`negative`
:data:`multiply`
:data:`divide`
:data:`power`
:data:`subtract`
:data:`true_divide`
:data:`floor_divide`
:data:`fmod`
:data:`mod`
:data:`modf`
:data:`remainder`

:func:`clip`
:data:`sqrt`
:data:`square`
:data:`absolute`
:data:`sign`
:data:`maximum`
:data:`minimum`
:data:`fmax`
:data:`fmin`

Sorting, searching, and counting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`argmax`
:func:`argmin`
:func:`count_nonzero`
:func:`nonzero`
:func:`flatnonzero`
:func:`where`

Statistics
~~~~~~~~~~

:func:`amin`
:func:`amax`

:func:`mean`
:func:`var`
:func:`std`

:func:`bincount`

Padding
~~~~~~~

:func:`pad`

External Functions
~~~~~~~~~~~~~~~~~~

:func:`scatter_add`

Other
~~~~~

:func:`asnumpy`
