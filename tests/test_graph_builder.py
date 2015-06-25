from unittest import TestCase
import numpy as np
from chainer import cuda, Variable, Function
from chainer.graph_builder import build_graph

if cuda.available:
    cuda.init()

class MockFunction(Function):
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

    def forward_cpu(self, xs):
        assert len(xs) == self.n_in
        return tuple(np.zeros((1, 2)).astype(np.float32)
                for _ in xrange(self.n_out))

    def backward_cpu(self, xs, gys):
        assert len(xs) == self.n_in
        assert len(gys) == self.n_out
        return tuple(np.zeros_like(xs).astype(np.float32)
                for _ in xrange(self.n_in))

def mock_function(xs, n_out):
    return MockFunction(len(xs), n_out)(*xs)

class TestGraphBuilder(TestCase):
    # x-splitter-x'-f-y-splitter-y'-g-z
    def setUp(self):
        self.x = Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = mock_function((self.x,), 1)
        self.z = mock_function((self.y,), 1)

    # x
    def test_head_variable(self):
        self.assertEqual(len(build_graph((self.x,))), 0)

    # x-splitter-x'-f-y
    def test_intermediate_variable(self):
        self.assertEqual(len(build_graph((self.y,))), 4)

    # x-splitter-x'-f-y-splitter-y'-g-z
    def test_tail_variable(self):
        self.assertEqual(len(build_graph((self.z,))), 8)

    def test_multiple_outputs(self):
        self.assertEqual(len(build_graph((self.x, self.y))), 4)

    def test_multiple_outputs2(self):
        self.assertEqual(len(build_graph((self.x, self.z))), 8)

    def test_multiple_outputs3(self):
        self.assertEqual(len(build_graph((self.y, self.z))), 8)

    def test_multiple_outputs4(self):
        self.assertEqual(len(build_graph((self.x, self.y, self.z))), 8)

class TestGraphBuilder2(TestCase):
    # x-splitter-x' -f-y1
    #           \
    #            x''-f-y2
    def setUp(self):
        self.x = Variable(np.zeros((1, 2)).astype(np.float32))
        self.y1 = mock_function((self.x,), 1)
        self.y2 = mock_function((self.x,), 1)

    def test_head_node(self):
        self.assertEqual(len(build_graph((self.x,))), 0)

    def test_tail_node(self):
        self.assertEqual(len(build_graph((self.y1,))), 4)

    def test_tail_node2(self):
        self.assertEqual(len(build_graph((self.y2,))), 4)

    def test_multiple_tails(self):
        self.assertEqual(len(build_graph((self.y1, self.y2))), 7)

class TestGraphBuilder3(TestCase):
    # x-splitter-x'-f-y1
    #                \
    #                 y2
    def setUp(self):
        self.x = Variable(np.zeros((1, 2)).astype(np.float32))
        self.y1, self.y2 = mock_function((self.x,), 2)

    def test_head_node(self):
        self.assertEqual(len(build_graph((self.x,))), 0)

    def test_tail_node(self):
        self.assertEqual(len(build_graph((self.y1,))), 4)

    def test_tail_node2(self):
        self.assertEqual(len(build_graph((self.y2,))), 4)

    def test_multiple_tails(self):
        self.assertEqual(len(build_graph((self.y1, self.y2))), 5)

class TestGraphBuilder4(TestCase):
    # x1-splitter-x1'-f-y
    #                /
    # x2-splitter-x2' 
    def setUp(self):
        self.x1 = Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = Variable(np.zeros((1, 2)).astype(np.float32))
        self.y  = mock_function((self.x1, self.x2), 1)

    def test_head_node(self):
        self.assertEqual(len(build_graph((self.x1,))), 0)

    def test_head_node(self):
        self.assertEqual(len(build_graph((self.x2,))), 0)

    def test_multiple_heads(self):
        self.assertEqual(len(build_graph((self.x1, self.x2))), 0)

    def test_tail_node(self):
        self.assertEqual(len(build_graph((self.y,))), 7)

class TestGraphBuilder5(TestCase):
    def setUp(self):
        self.x = Variable(np.zeros((1, 2)).astype(np.float32))
        self.y = 2 * self.x

    def test_tail_node(self):
        x_splitter = self.x.splitter()
        x_clone  = x_splitter.outputs[0]()
        f = self.y.creator

        edges = build_graph((self.y,))
        self.assertEqual(len(edges), 4)
        self.assertTrue((self.x, x_splitter) in edges)
        self.assertTrue((x_splitter, x_clone) in edges)
        self.assertTrue((f, self.y) in edges)

class TestGraphBuilder6(TestCase):
    def setUp(self):
        self.x1 = Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = Variable(np.zeros((1, 2)).astype(np.float32))
        self.y  = self.x1 + self.x2

    def test_tail_node(self):
        x1_splitter = self.x1.splitter()
        x2_splitter = self.x2.splitter()
        x1_clone = x1_splitter.outputs[0]()
        x2_clone = x2_splitter.outputs[0]()
        f = self.y.creator

        edges = build_graph((self.y,))
        self.assertEqual(len(edges), 7)
        self.assertTrue((self.x1, x1_splitter) in edges)
        self.assertTrue((x1_splitter, x1_clone) in edges)
        self.assertTrue((x1_clone, f) in edges)
        self.assertTrue((self.x2, x2_splitter) in edges)
        self.assertTrue((x2_splitter, x2_clone) in edges)
        self.assertTrue((x2_clone, f) in edges)
        self.assertTrue((f, self.y) in edges)

class TestGraphBuilder7(TestCase):
    def setUp(self):
        self.x1 = Variable(np.zeros((1, 2)).astype(np.float32))
        self.x2 = Variable(np.zeros((1, 2)).astype(np.float32))
        self.x3 = Variable(np.zeros((1, 2)).astype(np.float32))
        self.y  = 0.3 * (self.x1 + self.x2) + self.x3

    def test_tail_node(self):
        edges = build_graph((self.y,))
        self.assertEqual(len(edges), 18)
