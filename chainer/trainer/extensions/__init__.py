from chainer.trainer.extensions import _snapshot
from chainer.trainer.extensions import computational_graph
from chainer.trainer.extensions import evaluator
from chainer.trainer.extensions import exponential_decay
from chainer.trainer.extensions import linear_shift
from chainer.trainer.extensions import log_result


dump_graph = computational_graph.dump_graph
Evaluator = evaluator.Evaluator
ExponentialDecay = exponential_decay.ExponentialDecay
LinearShift = linear_shift.LinearShift
LogResult = log_result.LogResult
snapshot = _snapshot.snapshot
