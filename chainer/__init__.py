import collections
import os
import pkg_resources
import sys
import threading

from chainer import cuda  # NOQA
from chainer import dataset  # NOQA
from chainer import datasets  # NOQA
from chainer import flag  # NOQA
from chainer import function  # NOQA
from chainer import function_set  # NOQA
from chainer import functions  # NOQA
from chainer import initializer  # NOQA
from chainer import initializers  # NOQA
from chainer import iterators  # NOQA
from chainer import link  # NOQA
from chainer import links  # NOQA
from chainer import optimizer  # NOQA
from chainer import optimizers  # NOQA
from chainer import reporter  # NOQA
from chainer import serializer  # NOQA
from chainer import serializers  # NOQA
from chainer import training  # NOQA
from chainer import variable  # NOQA


# import class and function
from chainer.flag import AUTO  # NOQA
from chainer.flag import Flag  # NOQA
from chainer.flag import OFF  # NOQA
from chainer.flag import ON  # NOQA
from chainer.function import force_backprop_mode  # NOQA
from chainer.function import Function  # NOQA
from chainer.function import no_backprop_mode  # NOQA
from chainer.function_set import FunctionSet  # NOQA
from chainer.functions import array  # NOQA
from chainer.functions import basic_math  # NOQA
from chainer.initializer import Initializer  # NOQA
from chainer.initializers import init_weight  # NOQA
from chainer.link import Chain  # NOQA
from chainer.link import ChainList  # NOQA
from chainer.link import Link  # NOQA
from chainer.optimizer import GradientMethod  # NOQA
from chainer.optimizer import Optimizer  # NOQA
from chainer.reporter import DictSummary  # NOQA
from chainer.reporter import get_current_reporter  # NOQA
from chainer.reporter import report  # NOQA
from chainer.reporter import report_scope  # NOQA
from chainer.reporter import Reporter  # NOQA
from chainer.reporter import Summary  # NOQA
from chainer.serializer import AbstractSerializer  # NOQA
from chainer.serializer import Deserializer  # NOQA
from chainer.serializer import Serializer  # NOQA
from chainer.variable import Variable  # NOQA


if sys.version_info[:3] == (3, 5, 0):
    if not int(os.getenv('CHAINER_PYTHON_350_FORCE', '0')):
        msg = """
Chainer does not work with Python 3.5.0.

We strongly recommend to use another version of Python.
If you want to use Chainer with Python 3.5.0 at your own risk,
set 1 to CHAINER_PYTHON_350_FORCE environment variable."""

        raise Exception(msg)


__version__ = pkg_resources.get_distribution('chainer').version


thread_local = threading.local()


def get_function_hooks():
    if not hasattr(thread_local, 'function_hooks'):
        thread_local.function_hooks = collections.OrderedDict()
    return thread_local.function_hooks

_debug = False


def is_debug():
    """Get the debug mode.

    Returns:
        bool: Return ``True`` if Chainer is in debug mode.
    """
    return _debug


def set_debug(debug):
    """Set the debug mode.

    .. note::

        This method changes global state. When you use this method on
        multi-threading environment, it may affects other threads.

    Args:
        debug (bool): New debug mode.
    """
    global _debug
    _debug = debug


class DebugMode(object):
    """Debug mode context.

    This class provides a context manager for debug mode. When entering the
    context, it sets the debug mode to the value of `debug` parameter with
    memorizing its original value. When exiting the context, it sets the debug
    mode back to the original value.

    Args:
        debug (bool): Debug mode used in the context.
    """

    def __init__(self, debug):
        self._debug = debug

    def __enter__(self):
        self._old = is_debug()
        set_debug(self._debug)

    def __exit__(self, *_):
        set_debug(self._old)

basic_math.install_variable_arithmetics()
array.get_item.install_variable_get_item()

disable_experimental_feature_warning = False
