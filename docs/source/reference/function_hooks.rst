Function hooks
==============


Chainer provides a function-hook mechanism that enriches
the behavior of forward and backward propagation of :class:`~chainer.Function`.

.. module:: chainer.function

Base class
----------

.. autoclass:: FunctionHook
  :members:

.. module:: chainer.function_hooks

Concrete function hooks
-----------------------

.. autoclass:: PrintHook
  :members:

.. autoclass:: TimerHook
  :members:
