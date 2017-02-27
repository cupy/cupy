import heapq

from chainer import function
from chainer import variable


class DotNode(object):
    """Node of the computational graph, with utilities for dot language.

    This class represents a node of computational graph,
    with some utilities for dot language.

    Args:
        node: :class: `Variable` object or :class: `Function` object.
        attribute (dict): Attributes for the node.

    """

    def __init__(self, node, attribute=None):
        assert isinstance(node, (variable.Variable, function.Function))
        self.node = node
        self.id_ = id(node)
        self.attribute = {'label': node.label}
        if isinstance(node, variable.Variable):
            self.attribute.update({'shape': 'oval'})
        else:
            self.attribute.update({'shape': 'box'})
        if attribute is not None:
            self.attribute.update(attribute)

    @property
    def label(self):
        """The text that represents properties of the node.

        Returns:
            string: The text that represents the id and attributes of this
                node.
        """

        attributes = ["%s=\"%s\"" % (k, v) for (k, v)
                      in self.attribute.items()]
        return "%s [%s];" % (self.id_, ",".join(attributes))


class ComputationalGraph(object):

    """Class that represents computational graph.

    .. note::

      We assume that the computational graph is directed and acyclic.

    Args:
        nodes (list): List of nodes. Each node is either
             :class:`Variable` object or :class:`Function` object.
        edges (list): List of edges. Each edge consists of pair of nodes.
        variable_style (dict): Dot node style for variable.
        function_style (dict): Dot node style for function.
        rankdir (str): Direction of the graph that must be
            TB (top to bottom), BT (bottom to top), LR (left to right)
            or RL (right to left).

    """

    def __init__(self, nodes, edges, variable_style=None, function_style=None,
                 rankdir='TB'):
        self.nodes = nodes
        self.edges = edges
        self.variable_style = variable_style
        self.function_style = function_style
        if rankdir not in ('TB', 'BT', 'LR', 'RL'):
            raise ValueError('rankdir must be in TB, BT, LR or RL.')
        self.rankdir = rankdir

    def _to_dot(self):
        """Converts graph in dot format.

        `label` property of is used as short description of each node.
        Returns:
            str: The graph in dot format.

        """
        ret = 'digraph graphname{rankdir=%s;' % self.rankdir
        for node in self.nodes:
            assert isinstance(node, (variable.Variable, function.Function))
            if isinstance(node, variable.Variable):
                ret += DotNode(node, self.variable_style).label
            else:
                ret += DotNode(node, self.function_style).label
        for edge in self.edges:
            head, tail = edge
            if (isinstance(head, variable.Variable) and
                    isinstance(tail, function.Function)):
                head_attr = self.variable_style
                tail_attr = self.function_style
            elif (isinstance(head, function.Function) and
                  isinstance(tail, variable.Variable)):
                head_attr = self.function_style
                tail_attr = self.variable_style
            else:
                raise TypeError(
                    'head and tail should be the set of Variable and Function')
            head_node = DotNode(head, head_attr)
            tail_node = DotNode(tail, tail_attr)
            ret += "%s -> %s;" % (head_node.id_, tail_node.id_)
        ret += "}"
        return ret

    def dump(self, format='dot'):
        """Dumps graph as a text.

        Args:
            format(str): The graph language name of the output.
            Currently, it must be 'dot'.

        Returns:
            str: The graph in specified format.

        """
        if format == 'dot':
            return self._to_dot()
        else:
            NotImplementedError('Currently, only dot format is supported.')


def build_computational_graph(outputs, remove_split=True,
                              variable_style=None, function_style=None,
                              rankdir='TB'):
    """Builds a graph of functions and variables backward-reachable from outputs.

    Args:
        outputs(list): nodes from which the graph is constructed.
            Each element of outputs must be either :class:`Variable`
            object or :class:`Function` object.
        remove_split(bool): It must be ``True``. This argument is left for
            backward compatibility.
        variable_style(dict): Dot node style for variable.
            Possible keys are 'shape', 'color', 'fillcolor', 'style', and etc.
        function_style(dict): Dot node style for function.
        rankdir (str): Direction of the graph that must be
            TB (top to bottom), BT (bottom to top), LR (left to right)
            or RL (right to left).

    Returns:
        ComputationalGraph: A graph consisting of nodes and edges that
        are backward-reachable from at least one of ``outputs``.

        If ``unchain_backward`` was called in some variable in the
        computational graph before this function, backward step is
        stopped at this variable.

        For example, suppose that computational graph is as follows::

                |--> f ---> y
            x --+
                |--> g ---> z

        Let ``outputs = [y, z]``.
        Then the full graph is emitted.

        Next, let ``outputs = [y]``. Note that ``z`` and ``g``
        are not backward-reachable from ``y``.
        The resulting graph would be following::

            x ---> f ---> y

        See :class:`TestGraphBuilder` for details.

    """
    if not remove_split:
        raise ValueError('remove_split=False is not supported anymore')

    cands = []
    seen_edges = set()
    nodes = set()
    push_count = [0]

    # This class is for object that has not been implemented __eq__
    class HashableObject(object):

        def __init__(self, v):
            self.v = v

        def __hash__(self):
            return self.v.__hash__()

        def __eq__(self, r):
            return self.v is r.v

    def add_cand(cand):
        heapq.heappush(cands, (-cand.rank, push_count[0], cand))
        push_count[0] += 1

    for o in outputs:
        add_cand(o)
        nodes.add(HashableObject(o))

    while cands:
        _, _, cand = heapq.heappop(cands)
        if isinstance(cand, variable.Variable):
            creator = cand.creator
            if creator is not None and (creator, cand) not in seen_edges:
                add_cand(creator)
                seen_edges.add((creator, cand))
                nodes.add(HashableObject(creator))
                nodes.add(HashableObject(cand))
        elif isinstance(cand, function.Function):
            for input_ in cand.inputs:
                if input_ is not cand and (input_, cand) not in seen_edges:
                    add_cand(input_)
                    seen_edges.add((input_, cand))
                    nodes.add(HashableObject(input_))
                    nodes.add(HashableObject(cand))
    return ComputationalGraph(list(i.v for i in nodes), list(seen_edges),
                              variable_style, function_style, rankdir)
