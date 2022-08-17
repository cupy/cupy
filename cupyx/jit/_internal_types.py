from typing import Any, Optional, Union

from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules


class Expr:

    def __repr__(self) -> str:
        raise NotImplementedError


class Data(Expr):

    def __init__(self, code: str, ctype: _cuda_types.TypeBase) -> None:
        assert isinstance(code, str)
        assert isinstance(ctype, _cuda_types.TypeBase)
        self.code = code
        self.ctype = ctype
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
