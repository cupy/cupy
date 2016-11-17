import chainer
import inspect
import warnings


def experimental(api_name=None):
    """A function for marking APIs as experimental

    Developer of the API can mark it as *experimental* by calling
    this function. When users call experimental APIs, `FutureWarning`
    is raised once for each experimental API.
    The presentation of `FutureWarning` is disabled by setting
    `chainer.disable_experimental_warning` to `True`,
    which is `False` by default.

    Args:
        api_name(str): The name of an API marked as experimental.
            If it is `None`, it is inferred from the caller.
            If the caller is a function, its name is used.
            If the caller is a method or a class mehtod,
            `<class name>.<method name>` is used.
    """

    if api_name is None:
        api_name = inspect.stack()[1][3]

        frame = inspect.stack()[1][0]
        f_locals = frame.f_locals
        if 'self' in f_locals:
            class_name = f_locals['self'].__class__.__name__
            api_name = class_name + '.' + api_name
        elif 'cls' in f_locals:
            class_name = f_locals['cls'].__name__
            api_name = class_name + '.' + api_name

    if not chainer.disable_experimental_feature_warning:
        warnings.warn('{} is an experimental API. '
                      'The interface can change in the future'.format(
                          api_name),
                      FutureWarning)
