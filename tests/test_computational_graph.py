import unittest

import numpy as np
import six

from chainer import computational_graph as c
from chainer import function
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


class TestGraphBuilder(unittest.TestCase):
    # x-splitter-x'-f-y-splitter-y'-g-z
    def setUp(self):
        self.x = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = mock_function((self.x,), 1)
        self.z = mock_function((self.y,), 1)

    # x
    def test_head_variable(self):
        self.assertEqual(len(c.build_computational_graph((self.x,), False)), 0)
        self.assertEqual(len(c.build_computational_graph((self.x,),  True)), 0)

    def test_intermediate_variable(self):
        # x-splitter-x'-f-y
        self.assertEqual(len(c.build_computational_graph((self.y,), False)), 4)
        # x-f-y (splitter removed)
        self.assertEqual(len(c.build_computational_graph((self.y,),  True)), 2)

    def test_tail_variable(self):
        # x-splitter-x'-f-y-splitter-y'-g-z
        self.assertEqual(len(c.build_computational_graph((self.z,), False)), 8)
        # x-f-y-g-z (splitter removed)
        self.assertEqual(len(c.build_computational_graph((self.z,),  True)), 4)

    def test_multiple_outputs(self):
        edges = c.build_computational_graph((self.x, self.y), False)
        self.assertEqual(len(edges), 4)
        edges = c.build_computational_graph((self.x, self.y),  True)
        self.assertEqual(len(edges), 2)

    def test_multiple_outputs2(self):
        edges = c.build_computational_graph((self.x, self.z), False)
        self.assertEqual(len(edges), 8)
        edges = c.build_computational_graph((self.x, self.z),  True)
        self.assertEqual(len(edges), 4)

    def test_multiple_outputs3(self):
        edges = c.build_computational_graph((self.y, self.z), False)
        self.assertEqual(len(edges), 8)
        edges = c.build_computational_graph((self.y, self.z),  True)
        self.assertEqual(len(edges), 4)

    def test_multiple_outputs4(self):
        edges = c.build_computational_graph((self.x, self.y, self.z), False)
        self.assertEqual(len(edges), 8)
        edges = c.build_computational_graph((self.x, self.y, self.z),  True)
        self.assertEqual(len(edges), 4)


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
        self.assertEqual(len(c.build_computational_graph((self.x,), False)), 0)
        self.assertEqual(len(c.build_computational_graph((self.x,),  True)), 0)

    def test_tail_node(self):
        edges = c.build_computational_graph((self.y1,), False)
        self.assertEqual(len(edges), 4)
        edges = c.build_computational_graph((self.y1,),  True)
        self.assertEqual(len(edges), 2)

    def test_tail_node2(self):
        edges = c.build_computational_graph((self.y2,), False)
        self.assertEqual(len(edges), 4)
        edges = c.build_computational_graph((self.y2,),  True)
        self.assertEqual(len(edges), 2)

    def test_multiple_tails(self):
        edges = c.build_computational_graph((self.y1, self.y2), False)
        self.assertEqual(len(edges), 7)
        edges = c.build_computational_graph((self.y1, self.y2),  True)
        self.assertEqual(len(edges), 4)


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
        self.assertEqual(len(c.build_computational_graph((self.x,), False)), 0)
        self.assertEqual(len(c.build_computational_graph((self.x,),  True)), 0)

    def test_tail_node(self):
        edges = c.build_computational_graph((self.y1,), False)
        self.assertEqual(len(edges), 4)
        edges = c.build_computational_graph((self.y1,),  True)
        self.assertEqual(len(edges), 2)

    def test_tail_node2(self):
        edges = c.build_computational_graph((self.y2,), False)
        self.assertEqual(len(edges), 4)
        edges = c.build_computational_graph((self.y2,),  True)
        self.assertEqual(len(edges), 2)

    def test_multiple_tails(self):
        edges = c.build_computational_graph((self.y1, self.y2), False)
        self.assertEqual(len(edges), 5)
        edges = c.build_computational_graph((self.y1, self.y2),  True)
        self.assertEqual(len(edges), 3)


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
        edges = c.build_computational_graph((self.x1,), False)
        self.assertEqual(len(edges), 0)
        edges = c.build_computational_graph((self.x1,),  True)
        self.assertEqual(len(edges), 0)

    def test_head_node2(self):
        edges = c.build_computational_graph((self.x2,), False)
        self.assertEqual(len(edges), 0)
        edges = c.build_computational_graph((self.x2,),  True)
        self.assertEqual(len(edges), 0)

    def test_multiple_heads(self):
        edges = c.build_computational_graph((self.x1, self.x2), False)
        self.assertEqual(len(edges), 0)
        edges = c.build_computational_graph((self.x1, self.x2),  True)
        self.assertEqual(len(edges), 0)

    def test_tail_node(self):
        edges = c.build_computational_graph((self.y,), False)
        self.assertEqual(len(edges), 7)
        edges = c.build_computational_graph((self.y,),  True)
        self.assertEqual(len(edges), 3)


class TestGraphBuilder5(unittest.TestCase):
    def setUp(self):
        self.x = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = 2 * self.x

        self.x_splitter = self.x.splitter()
        self.x_clone = self.x_splitter.outputs[0]()
        self.f = self.y.creator

    def test_tail_node(self):
        edges = c.build_computational_graph((self.y,), False)
        self.assertEqual(len(edges), 4)
        self.assertTrue((self.x, self.x_splitter) in edges)
        self.assertTrue((self.x_splitter, self.x_clone) in edges)
        self.assertTrue((self.x_clone, self.f) in edges)
        self.assertTrue((self.f, self.y) in edges)

    def test_tail_node_remove_edge(self):
        edges = c.build_computational_graph((self.y,), True)
        self.assertEqual(len(edges), 2)
        self.assertTrue((self.x, self.f) in edges)
        self.assertTrue((self.f, self.y) in edges)


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

    def test_tail_node(self):
        edges = c.build_computational_graph((self.y,), False)
        self.assertEqual(len(edges), 7)
        self.assertTrue((self.x1, self.x1_splitter) in edges)
        self.assertTrue((self.x1_splitter, self.x1_clone) in edges)
        self.assertTrue((self.x1_clone, self.f) in edges)
        self.assertTrue((self.x2, self.x2_splitter) in edges)
        self.assertTrue((self.x2_splitter, self.x2_clone) in edges)
        self.assertTrue((self.x2_clone, self.f) in edges)
        self.assertTrue((self.f, self.y) in edges)

    def test_tail_node_remove_edge(self):
        edges = c.build_computational_graph((self.y,), True)
        self.assertEqual(len(edges), 3)
        self.assertTrue((self.x1, self.f) in edges)
        self.assertTrue((self.x2, self.f) in edges)
        self.assertTrue((self.f, self.y) in edges)


class TestGraphBuilder7(unittest.TestCase):
    def setUp(self):
        self.x1 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.x3 = variable.Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = 0.3 * (self.x1 + self.x2) + self.x3

    def test_tail_node(self):
        edges = c.build_computational_graph((self.y,), False)
        self.assertEqual(len(edges), 18)

    def test_tail_node_remove_edge(self):
        edges = c.build_computational_graph((self.y,), True)
        self.assertEqual(len(edges), 8)
