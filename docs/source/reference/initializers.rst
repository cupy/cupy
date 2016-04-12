Weight Initializers
===================

Weight initializer is an object that initializes the value of
:class:`numpy.ndarray` or :class:`cupy.ndarray`.
Weight initializers are instance of :class:`~chainer.Initializer`.
Each initializer are required to implement :meth:`~chainer.Initializer.__call__`
method that does initialization.
Typically, weight initializers are passed to `__init__` of :class:`~chainer.Link`
and initializes its the weights and biases.

.. module:: chainer.initializer


Base class
----------

.. autoclass:: Initializer
  :members:

.. module:: chainer.initializers


Helper function
---------------

.. autofunction:: init_weight

Concrete initializers
---------------------

.. autoclass:: Identity
  :members:

.. autoclass:: Constant
  :members:

.. autofunction:: Zero

.. autofunction:: One

.. autoclass:: Normal
  :members:

.. autoclass:: GlorotNormal
  :members:

.. autoclass:: HeNormal
  :members:

.. autoclass:: Orthogonal
  :members:

.. autoclass:: Uniform
  :members:

.. autoclass:: LeCunUniform
  :members:

.. autoclass:: GlorotUniform
  :members:

.. autoclass:: HeUniform
  :members:
