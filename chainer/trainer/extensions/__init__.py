from chainer.trainer.extensions import _snapshot
from chainer.trainer.extensions import computational_graph


dump_graph = computational_graph.dump_graph
snapshot = _snapshot.snapshot
