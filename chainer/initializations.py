from __future__ import absolute_import

import numpy as np

# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py


def init_weight(weights, initWith, scale=1,
                none_default=lambda shape: he_normal(shape, 1 / np.sqrt(2))):
    """Initializes the given weight matrix.

    Args:
            weights (~numpy.ndarray): Weight matrix to be initialized.
            initWith (value): The value to use to initialize the data. May be
                `None` (in which case the none_default value will be used), a
                scalar to set all values to, a matrix of the same shape to
                copy, or a callable that takes the shape of the matrix as input
                and returns a matrix of the correct size.
            scale (scalar): A constant to multiply initWith by, if initWith was
                a callable.
            none_default (value): Same as initWith, the value here must not be
                `None`, as it will be used as a default when initWith is
                `None`.

    """
    if initWith is None:
        initWith = none_default
    if callable(initWith):
        initWith = scale * initWith(weights.shape)
    if hasattr(initWith, 'shape'):
        # check needed for bilinear tests to pass
        assert weights.shape == initWith.shape
    weights[...] = initWith


def get_fans(shape):
    fan_in = np.prod(shape[1:])
    fan_out = shape[0]
    return fan_in, fan_out


def uniform(shape, scale=0.05):
    return np.random.uniform(low=-scale, high=scale, size=shape)


def normal(shape, scale=0.05):
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def lecun_uniform(shape, scale=1):
    '''Reference: LeCun 98, Efficient Backprop

    http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    '''
    fan_in, fan_out = get_fans(shape)
    s = scale * np.sqrt(3. / fan_in)
    return uniform(shape, s)


def glorot_normal(shape, scale=1):
    '''Reference: Glorot & Bengio, AISTATS 2010

    '''
    fan_in, fan_out = get_fans(shape)
    s = scale * np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s)


def glorot_uniform(shape, scale=1):
    fan_in, fan_out = get_fans(shape)
    s = scale * np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s)


def he_normal(shape, scale=1):
    '''Reference:  He et al., http://arxiv.org/abs/1502.01852

    '''
    fan_in, fan_out = get_fans(shape)
    s = scale * np.sqrt(2. / fan_in)
    return normal(shape, s)


def he_uniform(shape, scale=1):
    fan_in, fan_out = get_fans(shape)
    s = scale * np.sqrt(6. / fan_in)
    return uniform(shape, s)


def orthogonal(shape, scale=1.1):
    """From Lasagne.

    Reference: Saxe et al., http://arxiv.org/abs/1312.6120

    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q


def identity(shape, scale=1):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError('Identity matrix initialization can only be used '
                        'for 2D square matrices.')
    else:
        return scale * np.identity(shape[0])


def zero(shape):
    return np.zeros(shape)


def one(shape, scale=1):
    return scale * np.ones(shape)
