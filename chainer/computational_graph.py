import heapq

from chainer import function
from chainer import variable


class DotNode(object):
    """Node of the computational graph, with utilities for dot language.

    This class represents a node of computational graph,
    with some utilities for dot language.
    """

    def __init__(self, node):
        """Initializes DotNode.

        Args:
            node: :class: `Variable` object or :class: `Function` object.
        """

        assert isinstance(node, (variable.Variable, function.Function))
        self.node = node
        self.id_ = id(node)
        self.attribute = {
            "label": self.node.label,
            "shape": self._shape()
        }

    def _shape(self):
        """Returns shape type of node."""

        if isinstance(self.node, variable.Variable):
            return "oval"
        elif isinstance(self.node, function.Split):
            return "hexagon"
        else:
            return "box"

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
    """

    def __init__(self, nodes, edges):
        """Initializes computational graph.

        Args:
            nodes (list): List of nodes. Each node is either
                 :class:`Variable` object or :class:`Function` object.
            edges (list): List of edges. Each edge consists of pair of nodes.
        """
        self.nodes = nodes
        self.edges = edges

    def _to_dot(self):
        """Converts graph in dot format.

        `label` property of is used as short description of each node.
        Returns:
            str: The graph in dot format.
        """

        ret = "digraph graphname{"
        for node in self.nodes:
            assert isinstance(node, (variable.Variable, function.Function))
            ret += DotNode(node).label
        for edge in self.edges:
            head, tail = edge
            assert (isinstance(head, variable.Variable)
                    and isinstance(tail, function.Function)) or \
                   (isinstance(head, function.Function)
                    and isinstance(tail, variable.Variable))
            head_node = DotNode(head)
            tail_node = DotNode(tail)
            ret += "%s -> %s;" % (head_node.id_, tail_node.id_)
        ret += "}"
        return ret

    def dump(self, format='dot'):
        """Dumps graph as a text.

        Args
            format(str): The graph language name of the output.
            Currently, it must be 'dot'.

        Returns
            str: The graph in specified format.
        """
        if format == 'dot':
            return self._to_dot()
        else:
            NotImplementedError('Currently, only dot format is supported.')


def build_computational_graph(outputs, remove_split=True):
    """Builds a graph of functions and variables backward-reachable from outputs.

    Args:
        outputs(list): nodes from which the graph is constructed.
            Each element of outputs must be either :class:`Variable`
            object or :class:`Function` object.
        remove_split(bool): If it is ``True``, this function hides
            :class:`Split` functions and related variables from the graph.

    Returns:
        ComputationalGraph: A graph consisting of nodes and edges that
        are backward-reachable from at least one of ``outputs``.

        If ``unchain_backward`` was called in some variable in the
        computational graph before this function, backward step is
        stopped at this variable.

        For example, suppose that computational graph is as follows::

                                |--> x'  ---> f ---> y
            x ---> (splitter) --+
                                |--> x'' ---> g ---> z

        Let ``outputs = [y, z]``.
        If ``remove_split`` is ``False``, this method generates the graph
        itself. On the other hand, if ``remove_split`` is ``True``,
        ``splitter``, ``x'`` and ``x''`` are removed from the graph
        and ``x`` is directly connected to ``f`` and ``g``.
        Resulting graph will be::

               |--> f ---> y
            x -+
               |--> g ---> z

        Next, let ``outputs = [y]``. Note that ``z``, ``g``, and ``x''``
        are not backward-reachable from ``y``.
        If ``remove_split`` is ``False``, this function removes
        these unreachable nodes to get::

            x ---> (splitter) ---> x' ---> f ---> y

        If ``remove_split`` is ``True``, we further remove splitter
        and ``x'`` to get::

            x ---> f ---> y

        See :class:`TestGraphBuilder` for details.

    """

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
            if remove_split and isinstance(creator, function.Split):
                # assume that function.Split has only one input
                next_cand = creator.inputs[0]
                add_cand(next_cand)
                continue
            if creator is not None and (creator, cand) not in seen_edges:
                add_cand(creator)
                seen_edges.add((creator, cand))
                nodes.add(HashableObject(creator))
                nodes.add(HashableObject(cand))
        elif isinstance(cand, function.Function):
            if remove_split and isinstance(cand, function.Split):
                next_cand = creator.inputs[0]
                add_cand(next_cand)
                continue
            for input_ in cand.inputs:
                if input_ is not cand and (input_, cand) not in seen_edges:
                    creator = input_.creator
                    if remove_split and \
                       creator is not None and \
                       isinstance(creator, function.Split):
                        input_ = creator.inputs[0]
                    add_cand(input_)
                    seen_edges.add((input_, cand))
                    nodes.add(HashableObject(input_))
                    nodes.add(HashableObject(cand))
    return ComputationalGraph(list(i.v for i in nodes), list(seen_edges))
