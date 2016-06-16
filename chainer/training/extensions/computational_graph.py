import os

from chainer import computational_graph
from chainer.training import extension
from chainer import variable


_var_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
_func_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}


def dump_graph(root_name, out_name='cg.dot',
               variable_style=None, function_style=None):
    """Returns a trainer extension to dump a computational graph.

    This extension dumps a computational graph. The graph is output in DOT
    language.

    It only dumps a graph at the first iteration by default.

    Args:
        root_name (str): Name of the root of the computational graph. The
            root variable is retrieved by this name from the observation
            dictionary of the trainer.
        out_name (str): Output file name.
        variable_style (dict): Dot node style for variables. Each variable is
            rendered by an octagon by default.
        function_style (dict): Dot node style for functions. Each function is
            rendered by a rectangular by default.

    .. seealso::
       See :func:`~chainer.computational_graph.build_computational_graph`
       for the ``variable_style`` and ``function_style`` arguments.

    """
    def trigger(trainer):
        return trainer.updater.iteration == 1

    if variable_style is None:
        variable_style = _var_style
    if function_style is None:
        function_style = _func_style

    @extension.make_extension(trigger=trigger)
    def dump_graph(trainer):
        var = trainer.observation[root_name]
        if not isinstance(var, variable.Variable):
            raise TypeError('root value is not a Variable')
        cg = computational_graph.build_computational_graph(
            [var],
            variable_style=variable_style,
            function_style=function_style
        ).dump()

        out_path = os.path.join(trainer.out, out_name)
        # TODO(beam2d): support outputting images by the dot command
        with open(out_path, 'w') as f:
            f.write(cg)

    return dump_graph
