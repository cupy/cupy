.. module:: cupy.random

Random Sampling (``cupy.random``)
=================================

CuPy's random number generation routines are based on cuRAND.
They cover a small fraction of :mod:`numpy.random`.

The big difference of :mod:`cupy.random` from :mod:`numpy.random` is that :mod:`cupy.random` supports ``dtype`` option for most functions.
This option enables us to generate float32 values directly without any space overhead.


Sample random data
------------------

.. autofunction:: cupy.random.rand
.. autofunction:: cupy.random.randn
.. autofunction:: cupy.random.randint
.. autofunction:: cupy.random.random_integers
.. autofunction:: cupy.random.random_sample
.. autofunction:: cupy.random.random
.. autofunction:: cupy.random.ranf
.. autofunction:: cupy.random.sample


Distributions
-------------

.. autofunction:: cupy.random.lognormal
.. autofunction:: cupy.random.normal
.. autofunction:: cupy.random.standard_normal
.. autofunction:: cupy.random.uniform


Random number generator
-----------------------

.. autofunction:: cupy.random.seed
.. autofunction:: cupy.random.get_random_state
.. autoclass:: cupy.random.RandomState
   :members:
