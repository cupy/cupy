import collections
import os
import pkg_resources
import sys
import threading

from chainer import cuda  # NOQA
from chainer import dataset  # NOQA
from chainer import datasets  # NOQA
from chainer import flag
from chainer import function
from chainer import function_set
from chainer import functions  # NOQA
from chainer.functions import array
from chainer.functions import basic_math
from chainer import initializer
from chainer import initializers
from chainer import iterators  # NOQA
from chainer import link
from chainer import links  # NOQA
from chainer import optimizer
from chainer import optimizers  # NOQA
from chainer import reporter
from chainer import serializer
from chainer import serializers  # NOQA
from chainer import training  # NOQA
from chainer import variable


if sys.version_info[:3] == (3, 5, 0):
    if not int(os.getenv('CHAINER_PYTHON_350_FORCE', '0')):
        msg = """
Chainer does not work with Python 3.5.0.

We strongly recommend to use another version of Python.
If you want to use Chainer with Python 3.5.0 at your own risk,
set 1 to CHAINER_PYTHON_350_FORCE environment variable."""

        raise Exception(msg)


__version__ = pkg_resources.get_distribution('chainer').version

AbstractSerializer = serializer.AbstractSerializer
Chain = link.Chain
ChainList = link.ChainList
Deserializer = serializer.Deserializer
DictSummary = reporter.DictSummary
Flag = flag.Flag
force_backprop_mode = function.force_backprop_mode
Function = function.Function
no_backprop_mode = function.no_backprop_mode
FunctionSet = function_set.FunctionSet
GradientMethod = optimizer.GradientMethod
Link = link.Link
Optimizer = optimizer.Optimizer
Reporter = reporter.Reporter
Serializer = serializer.Serializer
Summary = reporter.Summary
Variable = variable.Variable
Initializer = initializer.Initializer

ON = flag.ON
OFF = flag.OFF
AUTO = flag.AUTO

get_current_reporter = reporter.get_current_reporter
report = reporter.report
report_scope = reporter.report_scope


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

init_weight = initializers.init_weight
