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

   cupy.random.beta
   cupy.random.binomial
   cupy.random.chisquare
   cupy.random.dirichlet
   cupy.random.exponential
   cupy.random.f
   cupy.random.gamma
   cupy.random.geometric
   cupy.random.gumbel
   cupy.random.hypergeometric
   cupy.random.laplace
   cupy.random.logistic
   cupy.random.lognormal
   cupy.random.logseries
   cupy.random.multinomial
   cupy.random.multivariate_normal
   cupy.random.normal
   cupy.random.pareto
   cupy.random.poisson
   cupy.random.rayleigh
   cupy.random.standard_cauchy
   cupy.random.standard_exponential
   cupy.random.standard_gamma
   cupy.random.standard_normal
   cupy.random.standard_t
   cupy.random.uniform
   cupy.random.vonmises
   cupy.random.wald
   cupy.random.weibull
   cupy.random.zipf


Random number generator
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.random.seed
   cupy.random.get_random_state
   cupy.random.set_random_state
   cupy.random.RandomState


Permutations
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.random.shuffle
   cupy.random.permutation
