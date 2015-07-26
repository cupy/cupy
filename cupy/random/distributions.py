from cupy.random import generator


# TODO(beam2d): Implement many distributions


def lognormal(mean=0.0, sigma=1.0, size=None):
    rs = generator.get_random_state()
    return rs.lognormal(mean, sigma, size=size)


def normal(loc=0.0, scale=1.0, size=None):
    rs = generator.get_random_state()
    return rs.normal(loc, scale, size=size)


def standard_normal(size=None):
    return normal(size=size)


def uniform(low=0.0, high=1.0, size=None):
    rs = generator.get_random_state()
    return rs.uniform(low, high, size=size)
