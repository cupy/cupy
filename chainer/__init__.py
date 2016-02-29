import collections
import threading
import pkg_resources

from chainer import flag
from chainer import function
from chainer import function_set
from chainer.functions import basic_math
from chainer import link
from chainer import optimizer
from chainer import serializer
from chainer import variable


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


class ThreadSafeOrderedDict(collections.OrderedDict):

    def __init__(self, *args, **kwargs):
        super(ThreadSafeOrderedDict, self).__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, type, value, traceback):
        self._lock.release()


global_function_hooks = ThreadSafeOrderedDict()

basic_math.install_variable_arithmetics()
