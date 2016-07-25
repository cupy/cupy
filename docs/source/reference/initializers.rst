Weight Initializers
===================

Weight initializer is an instance of :class:`~chainer.Initializer` that
destructively edits the contents of :class:`numpy.ndarray` or :class:`cupy.ndarray`.
Typically, weight initializers are passed to ``__init__`` of :class:`~chainer.Link`
and initializes its the weights and biases.

.. module:: chainer.initializer

Base class
----------

.. autoclass:: Initializer
  :members:


.. module:: chainer.initializers

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


.. module:: chainer

Helper function
---------------

.. autofunction:: init_weight
