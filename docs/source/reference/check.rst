Assertion and Testing
=====================

Chainer provides some facilities to make debugging easy.




.. _type-check-utils:

Type checking utilities
-----------------------
:class:`~chainer.Function` uses a systematic type checking of the :mod:`chainer.utils.type_check` module.
It enables users to easily find bugs of forward and backward implementations.
You can find examples of type checking in some function implementations.

.. automodule:: chainer.utils.type_check

.. autoclass:: Expr
   :members:
.. autofunction:: expect

.. autoclass:: TypeInfo
   :members:
.. autoclass:: TypeInfoTuple
   :members:

Gradient checking utilities
---------------------------
Most function implementations are numerically tested by *gradient checking*.
This method computes numerical gradients of forward routines and compares their results with the corresponding backward routines.
It enables us to make the source of issues clear when we hit an error of gradient computations.
The :mod:`chainer.gradient_check` module makes it easy to implement the gradient checking.

.. automodule:: chainer.gradient_check

.. autofunction:: check_backward
.. autofunction:: numerical_grad

Standard Assertions
-------------------
The assertions have same names as NumPy's ones.
The difference from NumPy is that they can accept both :class:`numpy.ndarray`
and :class:`cupy.ndarray`.

.. automodule:: chainer.testing
.. autofunction:: assert_allclose
