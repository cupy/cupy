import chainer
import inspect
import warnings


def experimental(api_name=None):
    """A function for marking APIs as experimental

    The developer of the API can mark it as *experimental* by calling
    this function. When users call experimental APIs, `FutureWarning`
    is raised once for each experimental API.
    The presentation of `FutureWarning` is disabled by setting
    `chainer.disable_experimental_warning` to `True`,
    which is `False` by default.

    The basic usage is to call it in the function or method we want to
    mark as experimental.

    >>> from chainer.utils import experiment
    ...
    ... def f(x):
    ...   experiment()
    ...   # concrete implementation of f follows
    ...
    ... f(1)
    FutureWarning: f is an experimental API. The interface can change in the future

    By default, if the caller is a function, its name is used as an API name,
    while if the caller is a method or a class mehtod,
    `<class name>.<method name>` is used.

    >>> class C(object):
    ...   def f(self):
    ...     experiment()
    ...
    ... c = C()
    ... c.f()
    FutureWarning: C.f is an experimental API. The interface can change in the future

    We can also define the name of experimental API name as an argument.
    It will use in the warning that will be issued.

    >>> def g(x):
    ...   experiment('My function')
    ...
    ... g(1)
    FutureWarning: g is an experimental API. The interface can change in the future

    If we want mark whole class, you should call this function
    in its `__init__` method.

    >>> class C():
    ...   def __init__(self):
    ...     experiment()

    >>> C()
    FutureWarning: C is an experimental API. The interface can change in the future
    
    If we want to mark `__init__` method only, rather than class itself.
    It is recommended that we explicitly feed its API name.

    >>> class D():
    ...   def __init__(self):
    ...     experiment('D.__init__')
    FutureWarning: D.__init__ is an experimental API. The interface can change in the future

    Args:
        api_name(str): The name of an API marked as experimental.
            If it is `None`, it is inferred from the caller by the following rule.
            * If the caller is a function, its name is used.
            * If the caller is `__init__` method of a class,
            The class name of the method is used.
            * If the caller is a method (other than `__init__`) of a class or a class mehtod,
            `<class name>.<method name>` is used.
    """

    if api_name is None:
        api_name = inspect.stack()[1][3]

        frame = inspect.stack()[1][0]
        f_locals = frame.f_locals
        if 'self' in f_locals:  # The caller is a method.
            class_name = f_locals['self'].__class__.__name__
            if api_name == '__init__':
                api_name = class_name
            else:
                api_name = class_name + '.' + api_name
        elif 'cls' in f_locals:  # The caller is class method.
            class_name = f_locals['cls'].__name__
            api_name = class_name + '.' + api_name

    if not chainer.disable_experimental_feature_warning:
        warnings.warn('{} is an experimental API. '
                      'The interface can change in the future'.format(
                          api_name),
                      FutureWarning)
