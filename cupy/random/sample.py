from cupy.random import distributions
from cupy.random import generator


def rand(*size):
    return random_sample(size=size)


def randn(*size):
    return distributions.normal(size=size)


def randint(low, high=None, size=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def random_integers(low, high=None, size=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def random_sample(size=None):
    rs = generator.get_random_state()
    return rs.random_sample(size=size)


def choice(a, size=None, replace=True, p=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError
