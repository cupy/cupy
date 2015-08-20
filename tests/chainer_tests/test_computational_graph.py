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


def _check(self, outputs, remove_split, node_num, edge_num):
    g = c.build_computational_graph(outputs, remove_split)
    self.assertEqual(len(g.nodes), node_num)
    self.assertEqual(len(g.edges), edge_num)


class TestGraphBuilder(unittest.TestCase):
    # x-splitter-x'-f-y-splitter-y'-g-z
    def setUp(self):
        self.x = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = mock_function((self.x,), 1)
        self.z = mock_function((self.y,), 1)

    # x
    def test_head_variable(self):
        _check(self, (self.x, ), False, 1, 0)
        _check(self, (self.x, ), True, 1, 0)

    def test_intermediate_variable(self):
        # x-splitter-x'-f-y
        _check(self, (self.y, ), False, 5, 4)
        # x-f-y (splitter removed)
        _check(self, (self.y, ), True, 3, 2)

    def test_tail_variable(self):
        # x-splitter-x'-f-y-splitter-y'-g-z
        _check(self, (self.z, ), False, 9, 8)
        # x-f-y-g-z (splitter removed)
        _check(self, (self.z, ), True, 5, 4)

    def test_multiple_outputs(self):
        _check(self, (self.x, self.y), False, 5, 4)
        _check(self, (self.x, self.y), True, 3, 2)

    def test_multiple_outputs2(self):
        _check(self, (self.x, self.z), False, 9, 8)
        _check(self, (self.x, self.z), True, 5, 4)

    def test_multiple_outputs3(self):
        _check(self, (self.y, self.z), False, 9, 8)
        _check(self, (self.y, self.z), True, 5, 4)

    def test_multiple_outputs4(self):
        _check(self, (self.x, self.y, self.z), False, 9, 8)
        _check(self, (self.x, self.y, self.z), True, 5, 4)


class TestGraphBuilder2(unittest.TestCase):
    # with splitter
    # x-splitter-x' -f-y1
    #           \
    #            x''-g-y2
    # without splitter
    # x-f-y1
    #  \
    #   g-y2
    def setUp(self):
        self.x = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y1 = mock_function((self.x,), 1)
        self.y2 = mock_function((self.x,), 1)

    def test_head_node(self):
        _check(self, (self.x, ), False, 1, 0)
        _check(self, (self.x, ), True, 1, 0)

    def test_tail_node(self):
        _check(self, (self.y1, ), False, 5, 4)
        _check(self, (self.y1, ), True, 3, 2)

    def test_tail_node2(self):
        _check(self, (self.y2, ), False, 5, 4)
        _check(self, (self.y2, ), True, 3, 2)

    def test_multiple_tails(self):
        _check(self, (self.y1, self.y2), False, 8, 7)
        _check(self, (self.y1, self.y2), True, 5, 4)


class TestGraphBuilder3(unittest.TestCase):
    # with splitter
    # x-splitter-x'-f-y1
    #                \
    #                 y2
    # without splitter
    # x-f-y1
    #    \
    #     y2
    def setUp(self):
        self.x = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y1, self.y2 = mock_function((self.x,), 2)

    def test_head_node(self):
        _check(self, (self.x, ), False, 1, 0)
        _check(self, (self.x, ), True, 1, 0)

    def test_tail_node(self):
        _check(self, (self.y1, ), False, 5, 4)
        _check(self, (self.y1, ), True, 3, 2)

    def test_tail_node2(self):
        _check(self, (self.y2, ), False, 5, 4)
        _check(self, (self.y2, ), True, 3, 2)

    def test_multiple_tails(self):
        _check(self, (self.y1, self.y2), False, 6, 5)
        _check(self, (self.y1, self.y2), True, 4, 3)


class TestGraphBuilder4(unittest.TestCase):
    # with splitter
    # x1-splitter-x1'-f-y
    #                /
    # x2-splitter-x2'
    # without splitter
    # x1-f-y
    #   /
    # x2
    def setUp(self):
        self.x1 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = mock_function((self.x1, self.x2), 1)

    def test_head_node1(self):
        _check(self, (self.x1, ), False, 1, 0)
        _check(self, (self.x1, ), True, 1, 0)

    def test_head_node2(self):
        _check(self, (self.x2, ), False, 1, 0)
        _check(self, (self.x2, ), True, 1, 0)

    def test_multiple_heads(self):
        _check(self, (self.x1, self.x2), False, 2, 0)
        _check(self, (self.x1, self.x2), True, 2, 0)

    def test_tail_node(self):
        _check(self, (self.y, ), False, 8, 7)
        _check(self, (self.y, ), True, 4, 3)


class TestGraphBuilder5(unittest.TestCase):
    def setUp(self):
        self.x = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = 2 * self.x

        self.x_splitter = self.x.splitter()
        self.x_clone = self.x_splitter.outputs[0]()
        self.f = self.y.creator

        self.g1 = c.build_computational_graph((self.y,), False)
        self.g2 = c.build_computational_graph((self.y,), True)

    def test_edges(self):
        self.assertEqual(len(self.g1.edges), 4)
        self.assertSetEqual(set(self.g1.edges),
                            set([(self.x, self.x_splitter),
                                 (self.x_splitter, self.x_clone),
                                 (self.x_clone, self.f),
                                 (self.f, self.y)]))

    def test_nodes(self):
        self.assertEqual(len(self.g1.nodes), 5)
        self.assertSetEqual(set(self.g1.nodes),
                            set([self.x,
                                 self.x_splitter,
                                 self.x_clone,
                                 self.f,
                                 self.y]))

    def test_edges_remove_split(self):
        self.assertEqual(len(self.g2.edges), 2)
        self.assertSetEqual(set(self.g2.edges),
                            set([(self.x, self.f),
                                 (self.f, self.y)]))

    def test_nodes_remove_split(self):
        self.assertEqual(len(self.g2.nodes), 3)
        self.assertSetEqual(set(self.g2.nodes),
                            set([self.x, self.f, self.y]))


class TestGraphBuilder6(unittest.TestCase):
    def setUp(self):
        self.x1 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = self.x1 + self.x2

        self.x1_splitter = self.x1.splitter()
        self.x2_splitter = self.x2.splitter()
        self.x1_clone = self.x1_splitter.outputs[0]()
        self.x2_clone = self.x2_splitter.outputs[0]()
        self.f = self.y.creator

        self.g1 = c.build_computational_graph((self.y,), False)
        self.g2 = c.build_computational_graph((self.y,), True)

    def test_edges(self):
        self.assertEqual(len(self.g1.edges), 7)
        self.assertSetEqual(set(self.g1.edges),
                            set([(self.x1, self.x1_splitter),
                                 (self.x1_splitter, self.x1_clone),
                                 (self.x1_clone, self.f),
                                 (self.x2, self.x2_splitter),
                                 (self.x2_splitter, self.x2_clone),
                                 (self.x2_clone, self.f),
                                 (self.f, self.y)]))

    def test_nodes(self):
        self.assertEqual(len(self.g1.nodes), 8)
        self.assertSetEqual(set(self.g1.nodes),
                            set([self.x1,
                                 self.x1_splitter,
                                 self.x1_clone,
                                 self.x2,
                                 self.x2_splitter,
                                 self.x2_clone,
                                 self.f,
                                 self.y]))

    def test_edges_remove_split(self):
        self.assertEqual(len(self.g2.edges), 3)
        self.assertSetEqual(set(self.g2.edges),
                            set([(self.x1, self.f),
                                 (self.x2, self.f),
                                 (self.f, self.y)]))

    def test_nodes_remove_split(self):
        self.assertEqual(len(self.g2.nodes), 4)
        self.assertSetEqual(set(self.g2.nodes),
                            set([self.x1,
                                 self.x2,
                                 self.f,
                                 self.y]))


class TestGraphBuilder7(unittest.TestCase):
    def setUp(self):
        self.x1 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x3 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = 0.3 * (self.x1 + self.x2) + self.x3

    def test_tail_node(self):
        _check(self, (self.y, ), False, 19, 18)
        _check(self, (self.y, ), True, 9, 8)


testing.run_module(__name__, __file__)
