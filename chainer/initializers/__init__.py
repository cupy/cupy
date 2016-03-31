import numpy

from chainer.initializers import constant
from chainer.initializers import normal
from chainer.initializers import orthogonal
from chainer.initializers import uniform


Identity = constant.Identity
Constant = constant.Constant
Zero = constant.Zero
One = constant.One
Normal = normal.Normal
GlorotNormal = normal.GlorotNormal
HeNormal = normal.HeNormal
Orthogonal = orthogonal.Orthogonal
Uniform = uniform.Uniform
LeCunUniform = uniform.LeCunUniform
GlorotUniform = uniform.GlorotUniform
HeUniform = uniform.HeUniform


def init_weight(weights, initWith, scale=1.0):
    """Initializes the given weight matrix.

    Args:
            weights (~numpy.ndarray): Weight matrix to be initialized.
            initWith (value): The value to use to initialize the data. May be
                `None` (in which case HeNormal is used as an initializer), a
                scalar to set all values to, a matrix of the same shape to
                copy, or a callable that takes the shape of the matrix as input
                and returns a matrix of the correct size.
            scale (scalar): A constant to multiply initWith by, if initWith was
                a callable.
    """

    if initWith is None:
        initWith = HeNormal(1 / numpy.sqrt(2))
    if callable(initWith):
        initWith = scale * initWith(weights.shape)
    if hasattr(initWith, 'shape'):
        # check needed for bilinear tests to pass
        assert weights.shape == initWith.shape
    weights[...] = initWith
