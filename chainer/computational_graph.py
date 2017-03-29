import heapq

from chainer import function
from chainer import variable

_var_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
_func_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}


class DotNode(object):
    """Node of the computational graph, with utilities for dot language.

    This class represents a node of computational graph,
    with some utilities for dot language.

    Args:
        node: :class: `Variable` object or :class: `Function` object.
        attribute (dict): Attributes for the node.
        show_name (bool): If `True`, the `name` attribute of the node is added
            to the label. Default is `True`.

    """

    def __init__(self, node, attribute=None, show_name=True):
        assert isinstance(node, (variable.Variable, function.Function))
        self.node = node
        self.id_ = id(node)
        self.attribute = {'label': node.label}
        if isinstance(node, variable.Variable):
            if show_name and hasattr(node, 'name') and node.name is not None:
                self.attribute['label'] = '{}: {}'.format(
                    node.name, self.attribute['label'])
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
        remove_variable (bool): If `True`, :class:`~chainer.Variable`s are
            removed from the resulting computational graph. Only
            :class:`~chainer.Function`s are shown in the output.
        show_name (bool): If `True`, the `name` attribute of each node is
            added to the label of the node. Default is `True`.

    .. note::

        The default behavior of :class:`~chainer.ComputationalGraph` has been
        changed from v1.23.0, so that it ouputs the richest representation of
        a graph as default, namely, styles are set and names of functions and
        variables are shown. To reproduce the same result as previous versions
        (<= v1.22.0), please specify `variable_style=None`,
        `function_style=None`, and `show_name=False` explicitly.

    """

    def __init__(self, nodes, edges, variable_style=_var_style,
                 function_style=_func_style, rankdir='TB',
                 remove_variable=False, show_name=True):
        self.nodes = nodes
        self.edges = edges
        self.variable_style = variable_style
        self.function_style = function_style
        if rankdir not in ('TB', 'BT', 'LR', 'RL'):
            raise ValueError('rankdir must be in TB, BT, LR or RL.')
        self.rankdir = rankdir
        self.remove_variable = remove_variable
        self.show_name = show_name

    def _to_dot(self):
        """Converts graph in dot format.

        `label` property of is used as short description of each node.
        Returns:
            str: The graph in dot format.

        """
        ret = 'digraph graphname{rankdir=%s;' % self.rankdir

        if self.remove_variable:
            self.nodes, self.edges = _skip_variable(self.nodes, self.edges)

        for node in self.nodes:
            assert isinstance(node, (variable.Variable, function.Function))
            if isinstance(node, variable.Variable):
                if not self.remove_variable:
                    ret += DotNode(
                        node, self.variable_style, self.show_name).label
            else:
                ret += DotNode(node, self.function_style, self.show_name).label

        drawn_edges = []
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
                if not self.remove_variable:
                    raise TypeError('head and tail should be the set of '
                                    'Variable and Function')
                else:
                    head_attr = self.function_style
                    tail_attr = self.function_style
            head_node = DotNode(head, head_attr, self.show_name)
            tail_node = DotNode(tail, tail_attr, self.show_name)
            edge = (head_node.id_, tail_node.id_)
            if edge in drawn_edges:
                continue
            ret += "%s -> %s;" % edge
            drawn_edges.append(edge)
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


def _skip_variable(nodes, edges):
    func_edges = []
    for edge_i, edge in enumerate(edges):
        head, tail = edge
        if isinstance(head, variable.Variable):
            if head.creator is not None:
                head = head.creator
            else:
                continue
        if isinstance(tail, variable.Variable):
            for node in nodes:
                if isinstance(node, function.Function):
                    for input_var in node.inputs:
                        if input_var is tail:
                            tail = node
                            break
                    if isinstance(tail, function.Function):
                        break
            else:
                continue
        func_edges.append((head, tail))
    return nodes, func_edges


def build_computational_graph(
        outputs, remove_split=True, variable_style=_var_style,
        function_style=_func_style, rankdir='TB', remove_variable=False,
        show_name=True):
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
        remove_variable (bool): If `True`, :class:`~chainer.Variable`s are
            removed from the resulting computational graph. Only
            :class:`~chainer.Function`s are shown in the output.
        show_name (bool): If `True`, the `name` attribute of each node is
            added to the label of the node. Default is `True`.

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

    .. note::

        The default behavior of :class:`~chainer.ComputationalGraph` has been
        changed from v1.23.0, so that it ouputs the richest representation of
        a graph as default, namely, styles are set and names of functions and
        variables are shown. To reproduce the same result as previous versions
        (<= v1.22.0), please specify `variable_style=None`,
        `function_style=None`, and `show_name=False` explicitly.

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
    return ComputationalGraph(
        list(i.v for i in nodes), list(seen_edges), variable_style,
        function_style, rankdir, remove_variable, show_name)
