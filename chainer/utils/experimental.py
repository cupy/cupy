import chainer
import warnings


def experimental(api_name):
    """Declares that user is using an experimental feature.

    The developer of an API can mark it as *experimental* by calling
    this function. When users call experimental APIs, :class:`FutureWarning`
    is issued.
    The presentation of :class:`FutureWarning` is disabled by setting
    ``chainer.disable_experimental_warning`` to ``True``,
    which is ``False`` by default.

    The basic usage is to call it in the function or method we want to
    mark as experimental along with the API name.

    .. testsetup::

        import sys
        import warnings

        warnings.simplefilter('always')

        def wrapper(message, category, filename, lineno, file=None, line=None):
            sys.stdout.write(warnings.formatwarning(
                message, category, filename, lineno))

        showwarning_orig = warnings.showwarning
        warnings.showwarning = wrapper

    .. testcleanup::

        warnings.showwarning = showwarning_orig

    .. testcode::

        from chainer import utils

        def f(x):
            utils.experimental('chainer.foo.bar.f')
            # concrete implementation of f follows

        f(1)

    .. testoutput::
        :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

        ... FutureWarning: chainer.foo.bar.f is experimental. \
The interface can change in the future. ...

    We can also make a whole class experimental. In that case,
    we should call this function in its ``__init__`` method.

    .. testcode::

        class C():
            def __init__(self):
              utils.experimental('chainer.foo.C')

        C()

    .. testoutput::
        :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

        ... FutureWarning: chainer.foo.C is experimental. \
The interface can change in the future. ...

    If we want to mark ``__init__`` method only, rather than class itself,
    it is recommended that we explicitly feed its API name.

    .. testcode::

        class D():
            def __init__(self):
                utils.experimental('D.__init__')

        D()

    .. testoutput::
        :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

        ...  FutureWarning: D.__init__ is experimental. \
The interface can change in the future. ...

    Currently, we do not have any sophisticated way to mark some usage of
    non-experimental function as experimental.
    But we can support such usage by explicitly branching it.

    .. testcode::

        def g(x, experimental_arg=None):
            if experimental_arg is not None:
                utils.experimental('experimental_arg of chainer.foo.g')

    Args:
        api_name(str): The name of an API marked as experimental.
    """

    if not chainer.disable_experimental_feature_warning:
        warnings.warn('{} is experimental. '
                      'The interface can change in the future.'.format(
                          api_name),
                      FutureWarning)
