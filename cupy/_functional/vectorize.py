import numpy

from cupy import core
from cupyx.jit import _interface
from cupyx.jit import _types


def _get_input_type(arg):
    if isinstance(arg, int):
        return 'l'
    if isinstance(arg, float):
        return 'd'
    if isinstance(arg, complex):
        return 'D'
    return arg.dtype.char


class vectorize(object):
    """Generalized function class.

    .. seealso:: :func:`numpy.vectorize`
    """

    def __init__(
            self, pyfunc, otypes=None, doc=None, excluded=None,
            cache=False, signature=None):
        """
        Args:
            pyfunc (callable): The target python funciton.
            otypes (str or list of dtypes, optional): The output data type.
            doc (str or None): The docstring for the function.
            excluded: Currently not supported.
            cache: Currently Ignored.
            signature: Currently not supported.
        """

        self.pyfunc = pyfunc
        self.__doc__ = doc or pyfunc.__doc__
        self.excluded = excluded
        self.cache = cache
        self.signature = signature
        self._kernel_cache = {}

        self.otypes = None
        if otypes is not None:
            self.otypes = ''.join([numpy.dtype(t).char for t in otypes])

        if excluded is not None:
            raise NotImplementedError(
                'cupy.vectorize does not support `excluded` option currently.')

        if signature is not None:
            raise NotImplementedError(
                'cupy.vectorize does not support `excluded` option currently.')

    def __call__(self, *args):
        itypes = ''.join([_get_input_type(x) for x in args])
        kern = self._kernel_cache.get(itypes, None)

        if kern is None:
            in_types = [_types.Scalar(t) for t in itypes]
            ret_type = None
            if self.otypes is not None:
                # TODO(asi1024): Implement
                raise NotImplementedError

            func = _interface._CudaFunction(self.pyfunc, 'numpy', device=True)
            code, ret_type = func._emit_code_from_types(in_types, ret_type)
            in_params = ', '.join(
                f'{t.dtype} in{i}' for i, t in enumerate(in_types))
            out_params = f'{ret_type.dtype} out0'
            body = 'out0 = {}({})'.format(
                func.name, ', '.join([f'in{i}' for i in range(len(in_types))]))
            kern = core.ElementwiseKernel(
                in_params, out_params, body, preamble=code)
            self._kernel_cache[itypes] = kern

        return kern(*args)
