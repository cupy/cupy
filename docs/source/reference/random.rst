.. module:: cupy.random

Random Sampling (``cupy.random``)
=================================

CuPy's random number generation routines are based on cuRAND.
They cover a small fraction of :mod:`numpy.random`.

The big difference of :mod:`cupy.random` from :mod:`numpy.random` is that :mod:`cupy.random` supports ``dtype`` option for most functions.
This option enables us to generate float32 values directly without any space overhead.


Sample random data
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.random.choice
   cupy.random.rand
   cupy.random.randn
   cupy.random.randint
   cupy.random.random_integers
   cupy.random.random_sample
   cupy.random.random
   cupy.random.ranf
   cupy.random.sample
   cupy.random.bytes


Distributions
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.random.gumbel
   cupy.random.lognormal
   cupy.random.normal
   cupy.random.standard_normal
   cupy.random.uniform


Random number generator
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.random.seed
   cupy.random.get_random_state
   cupy.random.RandomState
