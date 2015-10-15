import pkg_resources

from chainer import function
from chainer import function_set
from chainer.functions import basic_math
from chainer import optimizer
from chainer import serializer
from chainer import variable


__version__ = pkg_resources.get_distribution('chainer').version

AbstractSerializer = chainer.AbstractSerializer
Deserializer = chainer.Deserializer
Function = function.Function
FunctionSet = function_set.FunctionSet
Optimizer = optimizer.Optimizer
Serializer = chainer.Serializer
Variable = variable.Variable

basic_math.install_variable_arithmetics()
