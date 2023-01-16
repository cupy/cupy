Mathematical functions with automatic domain
============================================

.. Hint:: `NumPy API Reference: Mathematical functions with automatic domain <https://numpy.org/doc/stable/reference/routines.emath.html>`_

Wrapper functions to more user-friendly calling of certain math functions
whose output data-type is different than the input data-type in certain
domains of the input.
For example, for functions like :func:`cupy.log` with branch cuts, the versions in this
module provide the mathematically valid answers in the complex plane::

.. doctest::

   >>> import math
   >>> cupy.emath.log(-math.exp(1)) == (1+1j*math.pi)
   array(True)

Similarly, :func:`cupy.sqrt`, other base logarithms, :func:`cupy.power` and trig functions are
correctly handled.  See their respective docstrings for specific examples.

.. currentmodule:: cupy.lib.scimath

Functions
-----------------------

.. autosummary::
   :toctree: generated/

   sqrt
   log
   log2
   logn
   log10
   power
   arccos
   arcsin
   arctanh
