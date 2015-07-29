import numpy

from cupy.random import distributions
from cupy.random import generator
from cupy.random import sample


rand = sample.rand
randn = sample.randn
random_sample = sample.random_sample
random = random_sample
ranf = random_sample
sample = random_sample
bytes = numpy.random.bytes

lognormal = distributions.lognormal
normal = distributions.normal
standard_normal = distributions.standard_normal
uniform = distributions.uniform

RandomState = generator.RandomState
seed = generator.seed
set_float_type = generator.set_float_type
reset_states = generator.reset_states
