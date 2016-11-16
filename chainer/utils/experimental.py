import chainer
import inspect
import warnings


def experimental(api_name=None):

    if api_name is None:
        api_name = inspect.stack()[1][3]

        frame = inspect.stack()[1][0]
        f_locals = frame.f_locals
        if 'self' in f_locals:
            class_name = f_locals['self'].__class__.__name__
            api_name = class_name + '.' + api_name

    if not chainer.disable_experimental_feature_warning:
        warnings.warn('{} is an experimental API. '
                      'The interface can change in the future'.format(
                          api_name),
                      FutureWarning)
