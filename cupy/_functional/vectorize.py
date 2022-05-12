import numpy

from cupy import _core
from cupyx.jit import _interface
from cupyx.jit import _cuda_types


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

    .. seealso:: :class:`numpy.vectorize`
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
                'cupy.vectorize does not support `signature`'
                ' option currently.')

    @staticmethod
    def _parse_out_param(return_type):
        if isinstance(return_type, _cuda_types.Scalar):
            dtypes = [return_type.dtype]
            out_lval = 'out0'
        elif isinstance(return_type, _cuda_types.Tuple):
            dtypes = []
            for t in return_type.types:
                if not isinstance(t, _cuda_types.Scalar):
                    raise TypeError(f'Invalid return type: {return_type}')
                dtypes.append(t.dtype)
            out_lvals = ', '.join([f'out{i}' for i in range(len(dtypes))])
            out_lval = f'thrust::tie({out_lvals})'
        else:
            raise TypeError(f'Invalid return type: {return_type}')

        out_params = [f'{dtype} out{i}' for i, dtype in enumerate(dtypes)]
        return ', '.join(out_params), out_lval

    def __call__(self, *args):
        itypes = ''.join([_get_input_type(x) for x in args])
        kern = self._kernel_cache.get(itypes, None)

        if kern is None:
            in_types = [_cuda_types.Scalar(t) for t in itypes]
            ret_type = None
            if self.otypes is not None:
                # TODO(asi1024): Implement
                raise NotImplementedError

            func = _interface._CudaFunction(self.pyfunc, 'numpy', device=True)
            result = func._emit_code_from_types(in_types, ret_type)
            in_params = ', '.join(
                f'{t.dtype} in{i}' for i, t in enumerate(in_types))
            in_args = ', '.join([f'in{i}' for i in range(len(in_types))])
            out_params, out_lval = self._parse_out_param(result.return_type)
            body = '{} = {}({})'.format(out_lval, result.func_name, in_args)
            # note: we don't worry about -D not working on ROCm here, because
            # we unroll all headers for HIP and so thrust::tuple et al are all
            # defined regardless if CUPY_JIT_MODE is defined or not
            kern = _core.ElementwiseKernel(
                in_params, out_params, body, 'cupy_vectorize',
                preamble=result.code,
                options=('-DCUPY_JIT_MODE', '--std=c++14'),
            )
            self._kernel_cache[itypes] = kern

        return kern(*args)
