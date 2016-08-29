from chainer import cuda
from chainer.functions.math import identity
from chainer import link


class Parameter(link.Link):

    """Link that just holds a parameter and returns it.

    .. deprecated:: v1.5
       The parameters are stored as variables as of v1.5. Use them directly
       instead.

    Args:
        array: Initial parameter array.

    Attributes:
        W (~chainer.Variable): Parameter variable.

    """

    def __init__(self, array):
        super(Parameter, self).__init__()
        self.add_param('W', array.shape, dtype=array.dtype)
        self.W.data = array
        if isinstance(array, cuda.ndarray):
            self.to_gpu(array)

    def __call__(self, volatile='off'):
        """Returns the parameter variable.

        Args:
            volatile (~chainer.Flag): The volatility of the returned variable.

        Returns:
            ~chainer.Variable: A copy of the parameter variable with given
            volatility.

        """
        # The first identity creates a copy of W, and the second identity cuts
        # the edge if volatility is ON
        W = identity.identity(self.W)
        W.volatile = volatile
        return identity.identity(W)
