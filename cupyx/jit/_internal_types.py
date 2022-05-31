import itertools

from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules


class Expr:
    pass


class Data(Expr):
    def __init__(self, code: str, ctype: _cuda_types.TypeBase):
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

    def __repr__(self):
        return f'<Data code = "{self.code}", type = {self.ctype}>'

    @classmethod
    def init(cls, x: Expr, env):
        if isinstance(x, Data):
            return x
        if isinstance(x, Constant):
            ctype = _cuda_typerules.get_ctype_from_scalar(env.mode, x.obj)
            code = _cuda_types.get_cuda_code_from_constant(x.obj, ctype)
            return Data(code, ctype)
        raise TypeError(f"'{x}' cannot be interpreted as a cuda object.")


class Constant(Expr):
    def __init__(self, obj):
        self._obj = obj

    @property
    def obj(self):
        return self._obj

    def __repr__(self):
        return f'<Constant obj = "{self.obj}">'


class Range(Expr):

    def __init__(self, start, stop, step, ctype, step_is_positive, *,
                 unroll=None):
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

    def call(self, env, *args, **kwargs):
        for x in itertools.chain(args, kwargs.values()):
            if not isinstance(x, Constant):
                raise TypeError('Arguments must be constants.')
        args = [x.obj for x in args]
        kwargs = dict([(k, v.obj) for k, v in kwargs.items()])
        return self.call_const(env, *args, **kwargs)

    def call_const(self, env, *args, **kwarg):
        raise NotImplementedError

    def __init__(self):
        self.__doc__ = type(self).__call__.__doc__

    def __call__(self):
        raise RuntimeError('Cannot call this function from Python layer.')

    @classmethod
    def from_class_method(cls, instance_name, method):
        # - this helper wraps every class method as a BuiltinFunc
        # - method must return a valid Expr
        # TODO(leofang): if performance is concerned, we could cache _Wrapper
        # for each method.__func__, and overwrite with the provided instance
        class _Wrapper(BuiltinFunc):

            def call(self, env, *args, **kwargs):
                data = method(env, *args, **kwargs)
                return Data(f'{instance_name}.{data.code}', data.ctype)

        return _Wrapper()
