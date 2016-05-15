import os

from chainer import computational_graph
from chainer.training import extension
from chainer import variable


def dump_graph(root_name, out_name='cg.dot'):
    """Returns a trainer extension to dump a computational graph.

    This extension dumps a computational graph. The graph is output in DOT
    language.

    It only dumps a graph at the first iteration by default.

    Args:
        root_name (str): Name of the root of the computational graph. The
            root variable is retrieved by this name from the observation
            dictionary of the trainer.
        out_name (str): Output file name.

    """
    def trigger(trainer):
        return trainer.updater.iteration == 1

    @extension.make_extension(name='dump_graph', trigger=trigger)
    def ext(trainer):
        var = trainer.observation[root_name]
        if not isinstance(variable.Variable):
            raise TypeError('root value is not a Variable')
        cg = computational_graph.build_computational_graph([var]).dump()

        out_path = os.path.join(trainer.out, out_name)
        # TODO(beam2d): support outputting images by the dot command
        with open(out_path, 'w') as f:
            f.write(cg)

    return ext
