import unittest

import numpy as np
import six

from chainer import computational_graph as c
from chainer import function
from chainer import testing
from chainer import variable


class MockFunction(function.Function):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

    def forward_cpu(self, xs):
        assert len(xs) == self.n_in
        return tuple(np.zeros((1, 2)).astype(np.float32)
                     for _ in six.moves.range(self.n_out))

    def backward_cpu(self, xs, gys):
        assert len(xs) == self.n_in
        assert len(gys) == self.n_out
        return tuple(np.zeros_like(xs).astype(np.float32)
                     for _ in six.moves.range(self.n_in))


def mock_function(xs, n_out):
    return MockFunction(len(xs), n_out)(*xs)


def _check(self, outputs, node_num, edge_num):
    g = c.build_computational_graph(outputs)
    self.assertEqual(len(g.nodes), node_num)
    self.assertEqual(len(g.edges), edge_num)


class TestGraphBuilder(unittest.TestCase):

    # x-f-y-g-z
    def setUp(self):
        self.x = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = mock_function((self.x,), 1)
        self.z = mock_function((self.y,), 1)

    # x
    def test_head_variable(self):
        _check(self, (self.x, ), 1, 0)

    def test_intermediate_variable(self):
        # x-f-y
        _check(self, (self.y, ), 3, 2)

    def test_tail_variable(self):
        # x-f-y-g-z
        _check(self, (self.z, ), 5, 4)

    def test_multiple_outputs(self):
        _check(self, (self.x, self.y), 3, 2)

    def test_multiple_outputs2(self):
        _check(self, (self.x, self.z), 5, 4)

    def test_multiple_outputs3(self):
        _check(self, (self.y, self.z), 5, 4)

    def test_multiple_outputs4(self):
        _check(self, (self.x, self.y, self.z), 5, 4)


class TestGraphBuilder2(unittest.TestCase):

    # x-f-y1
    #  \
    #   g-y2
    def setUp(self):
        self.x = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y1 = mock_function((self.x,), 1)
        self.y2 = mock_function((self.x,), 1)

    def test_head_node(self):
        _check(self, (self.x, ), 1, 0)

    def test_tail_node(self):
        _check(self, (self.y1, ), 3, 2)

    def test_tail_node2(self):
        _check(self, (self.y2, ), 3, 2)

    def test_multiple_tails(self):
        _check(self, (self.y1, self.y2), 5, 4)


class TestGraphBuilder3(unittest.TestCase):

    # x-f-y1
    #    \
    #     y2
    def setUp(self):
        self.x = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y1, self.y2 = mock_function((self.x,), 2)

    def test_head_node(self):
        _check(self, (self.x, ), 1, 0)

    def test_tail_node(self):
        _check(self, (self.y1, ), 3, 2)

    def test_tail_node2(self):
        _check(self, (self.y2, ), 3, 2)

    def test_multiple_tails(self):
        _check(self, (self.y1, self.y2), 4, 3)


class TestGraphBuilder4(unittest.TestCase):

    # x1-f-y
    #   /
    # x2
    def setUp(self):
        self.x1 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = mock_function((self.x1, self.x2), 1)

    def test_head_node1(self):
        _check(self, (self.x1, ), 1, 0)

    def test_head_node2(self):
        _check(self, (self.x2, ), 1, 0)

    def test_multiple_heads(self):
        _check(self, (self.x1, self.x2), 2, 0)

    def test_tail_node(self):
        _check(self, (self.y, ), 4, 3)


class TestGraphBuilder5(unittest.TestCase):

    def setUp(self):
        self.x = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = 2 * self.x
        self.f = self.y.creator
        self.g = c.build_computational_graph((self.y,))

    def test_edges(self):
        self.assertEqual(len(self.g.edges), 2)
        self.assertSetEqual(set(self.g.edges),
                            {(self.x, self.f), (self.f, self.y)})

    def test_nodes(self):
        self.assertEqual(len(self.g.nodes), 3)
        self.assertSetEqual(set(self.g.nodes),
                            {self.x, self.f, self.y})


class TestGraphBuilder6(unittest.TestCase):

    def setUp(self):
        self.x1 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = self.x1 + self.x2
        self.f = self.y.creator
        self.g = c.build_computational_graph((self.y,))

    def test_edges(self):
        self.assertEqual(len(self.g.edges), 3)
        self.assertSetEqual(set(self.g.edges),
                            {(self.x1, self.f),
                             (self.x2, self.f),
                             (self.f, self.y)})

    def test_nodes(self):
        self.assertEqual(len(self.g.nodes), 4)
        self.assertSetEqual(set(self.g.nodes),
                            {self.x1, self.x2, self.f, self.y})


class TestGraphBuilder7(unittest.TestCase):

    def setUp(self):
        self.x1 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x3 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = 0.3 * (self.x1 + self.x2) + self.x3

    def test_tail_node(self):
        _check(self, (self.y, ), 9, 8)


class TestGraphBuilderStylization(unittest.TestCase):

    def setUp(self):
        self.x1 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = self.x1 + self.x2
        self.f = self.y.creator
        self.variable_style = {'label': 'variable_0', 'shape': 'octagon',
                               'style': 'filled', 'fillcolor': '#E0E0E0'}
        self.function_style = {'label': 'function_0', 'shape': 'record',
                               'style': 'filled', 'fillcolor': '#6495ED'}
        self.g = c.build_computational_graph(
            (self.y,), variable_style=self.variable_style,
            function_style=self.function_style)

    def test_dotfile_content(self):
        dotfile_content = self.g.dump()
        for style in [self.variable_style, self.function_style]:
            for key, value in style.items():
                self.assertIn('{0}="{1}"'.format(key, value), dotfile_content)


class TestGraphBuilderRankdir(unittest.TestCase):

    def setUp(self):
        self.x1 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = self.x1 + self.x2

    def test_randir(self):
        for rankdir in ['TB', 'BT', 'LR', 'RL']:
            g = c.build_computational_graph((self.y,), rankdir=rankdir)
            self.assertIn('rankdir=%s' % rankdir, g.dump())

    def test_randir_invalid(self):
        self.assertRaises(ValueError,
                          c.build_computational_graph, (self.y,), rankdir='TL')


testing.run_module(__name__, __file__)
