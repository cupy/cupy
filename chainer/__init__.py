import collections
import pkg_resources
import sys
import threading
import warnings

from chainer import flag
from chainer import function
from chainer import function_set
from chainer.functions import basic_math
from chainer import link
from chainer import optimizer
from chainer import serializer
from chainer import variable


if sys.version_info[:3] == (3, 5, 0):
    warnings.warn('Python 3.5.0 is not recommended. Use newer version.')


__version__ = pkg_resources.get_distribution('chainer').version

AbstractSerializer = serializer.AbstractSerializer
Chain = link.Chain
ChainList = link.ChainList
Deserializer = serializer.Deserializer
Flag = flag.Flag
Function = function.Function
FunctionSet = function_set.FunctionSet
GradientMethod = optimizer.GradientMethod
Link = link.Link
Optimizer = optimizer.Optimizer
Serializer = serializer.Serializer
Variable = variable.Variable

ON = flag.ON
OFF = flag.OFF
AUTO = flag.AUTO


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

    note::

        This method changes global state. When you use this method on
        multi-threading environment, it may affects other threads.

    Args:
        debug (bool): New debug mode.
    """
    global _debug
    _debug = debug

basic_math.install_variable_arithmetics()
