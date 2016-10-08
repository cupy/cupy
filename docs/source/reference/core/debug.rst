Debug mode
==========

In debug mode, Chainer checks values of variables on runtime and shows more
detailed error messages.
It helps you to debug your programs.
Instead it requires additional overhead time.

In debug mode, Chainer checks all results of forward and backward computation, and if it founds a NaN value, it raises :class:`RuntimeError`.
Some functions and links also check validity of input values.

.. currentmodule:: chainer

.. autofunction:: is_debug
.. autofunction:: set_debug
.. autoclass:: DebugMode
