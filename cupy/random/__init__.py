import numpy

from cupy.random import distributions
from cupy.random import generator
from cupy.random import sample as sample_


rand = sample_.rand
randn = sample_.randn
random_sample = sample_.random_sample
random = random_sample
ranf = random_sample
sample = random_sample
bytes = numpy.random.bytes

lognormal = distributions.lognormal
normal = distributions.normal
standard_normal = distributions.standard_normal
uniform = distributions.uniform

RandomState = generator.RandomState
get_random_state = generator.get_random_state
seed = generator.seed
reset_states = generator.reset_states
