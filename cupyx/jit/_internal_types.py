import functools
import itertools
from typing import Any, NoReturn, Optional, Union, TYPE_CHECKING

from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules

if TYPE_CHECKING:
    from cupyx.jit._compile import Environment


class Expr:

    def __repr__(self) -> str:
        raise NotImplementedError


class Data(Expr):

    def __init__(self, code: str, ctype: _cuda_types.TypeBase) -> None:
        assert isinstance(code, str)
        assert isinstance(ctype, _cuda_types.TypeBase)
        self.code = code
        self.ctype = ctype
        if not isinstance(ctype, _cuda_types.Unknown):
            try:
                self.__doc__ = f'{str(ctype)} {code}\n{ctype.__doc__}'
            except NotImplementedError:
                self.__doc__ = f'{code}'

    @property
    def obj(self):
        raise ValueError(f'Constant value is requried: {self.code}')

    def __repr__(self) -> str:
        return f'<Data code = "{self.code}", type = {self.ctype}>'

    @classmethod
    def init(cls, x: Expr, env) -> 'Data':
        if isinstance(x, Data):
            return x
        if isinstance(x, Constant):
            if isinstance(x.obj, tuple):
                elts = [Data.init(Constant(e), env) for e in x.obj]
                elts_code = ', '.join([e.code for e in elts])
                if len(elts) == 2:
                    return Data(
                        f'thrust::make_pair({elts_code})',
                        _cuda_types.Tuple([x.ctype for x in elts]))
                return Data(
                    f'thrust::make_tuple({elts_code})',
                    _cuda_types.Tuple([x.ctype for x in elts]))
            ctype = _cuda_typerules.get_ctype_from_scalar(env.mode, x.obj)
            code = _cuda_types.get_cuda_code_from_constant(x.obj, ctype)
            return Data(code, ctype)
        raise TypeError(f"'{x}' cannot be interpreted as a cuda object.")


class Constant(Expr):

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    @property
    def obj(self) -> Any:
        return self._obj

    def __repr__(self) -> str:
        return f'<Constant obj = "{self.obj}">'


class Range(Expr):

    def __init__(
            self, start: Data, stop: Data, step: Data,
            ctype: _cuda_types.Scalar,
            step_is_positive: Optional[bool],
            *,
            unroll: Union[None, int, bool] = None,
    ) -> None:
        self.start = start
        self.stop = stop
        self.step = step
        self.ctype = ctype
        self.step_is_positive = step_is_positive  # True, False or None
        self.unroll = unroll


class BuiltinFunc(Expr):
    # subclasses must implement:
    # - either call or call_const
    # - `__call__` with a correct signature, which calls the parent's __call__

    def call(self, env: 'Environment', *args, **kwargs) -> Expr:
        for x in itertools.chain(args, kwargs.values()):
            if not isinstance(x, Constant):
                raise TypeError('Arguments must be constants.')
        args = tuple([x.obj for x in args])
        kwargs = dict([(k, v.obj) for k, v in kwargs.items()])
        return self.call_const(env, *args, **kwargs)

    def call_const(self, env: 'Environment', *args: Any, **kwarg: Any) -> Expr:
        raise NotImplementedError

    def __init__(self) -> None:
        self.__doc__ = type(self).__call__.__doc__

    def __call__(self) -> NoReturn:
        raise RuntimeError('Cannot call this function from Python layer.')

    def __repr__(self) -> str:
        return '<cupyx.jit function>'

    @classmethod
    def from_class_method(cls, method, ctype_self, instance):
        class _Wrapper(BuiltinFunc):

            def call(self, env, *args, **kwargs):
                return method(ctype_self, env, instance, *args)

        return _Wrapper()


def wraps_class_method(method):

    @functools.wraps(method)
    def f(ctype_self: _cuda_types.TypeBase, instance: Data) -> BuiltinFunc:
        return BuiltinFunc.from_class_method(method, ctype_self, instance)

    return f
