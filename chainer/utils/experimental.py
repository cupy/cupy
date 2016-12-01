import chainer
import warnings


def experimental(api_name=None):
    """A function for marking APIs as experimental.

    The developer of an API can mark it as *experimental* by calling
    this function. When users call experimental APIs, :class:`FutureWarning`
    is issued.
    The presentation of :class:`FutureWarning` is disabled by setting
    ``chainer.disable_experimental_warning`` to ``True``,
    which is ``False`` by default.

    The basic usage is to call it in the function or method we want to
    mark as experimental along with the API name.

    >>> from chainer.utils import experimental
    ...
    ... def f(x):
    ...   experimental('chainer.foo.bar.f')
    ...   # concrete implementation of f follows
    ...
    ... f(1)
    FutureWarning: chainer.experimental.f is an experimental API. \
The interface can change in the future.

    We can also make a whole class experimental. In that case,
    we should call this function in its `__init__` method.

    >>> class C():
    ...   def __init__(self):
    ...     experimental('chainer.foo.C')
    ...
    ... C()
    FutureWarning: chainer.foo.C is an experimental API. \
The interface can change in the future

    If we want to mark ``__init__`` method only, rather than class itself.
    It is recommended that we explicitly feed its API name.

    >>> class D():
    ...   def __init__(self):
    ...     experimental('D.__init__')
    ...
    ... D()
    FutureWarning: D.__init__ is an experimental API. \
The interface can change in the future

    Currently, we do not have any sophisticated way to mark some usage of
    non-experimental function as experimental.
    But we can support such usage by explicitly branching it.

    >>> def g(x, experimental_arg=None):
    ...   if experimental_arg is not None:
    ...     experimental('experimental_arg of chainer.foo.g')

    Args:
        api_name(str): The name of an API marked as experimental.
    """

    if not chainer.disable_experimental_feature_warning:
        warnings.warn('{} is experimental. '
                      'The interface can change in the future.'.format(
                          api_name),
                      FutureWarning)
