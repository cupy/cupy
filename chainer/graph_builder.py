import heapq
from chainer import Variable, Function
from chainer.function import Split
import chainer.functions.basic_math

def build_graph(outputs, remove_split=False):
    cands = []
    seen_edges = set()

    for o in outputs:
        heapq.heappush(cands, (-o.rank, len(seen_edges), o))

    while cands:
        _, _, cand = heapq.heappop(cands)
        if isinstance(cand, Variable):
            creator = cand.creator
            if remove_split and isinstance(creator, Split):
                next_cand = creator.inputs[0] # assume that Split has only one input
                heapq.heappush(cands, (-next_cand.rank, len(seen_edges), next_cand))
                continue
            if creator is not None and (creator, cand) not in seen_edges:
                heapq.heappush(cands, (-creator.rank, len(seen_edges), creator))
                seen_edges.add((creator, cand))
        elif isinstance(cand, Function):
            if remove_split and isinstance(cand, Split):
                next_cand = creator.inputs[0]
                heapq.heappush(cands, (-next_cand.rank, len(seen_edges), next_cand))
                continue
            for input_ in cand.inputs:
                if input_ != cand and (input_, cand) not in seen_edges:
                    creator = input_.creator
                    if creator is not None and remove_split and isinstance(creator, Split):
                        input_ = creator.inputs[0]
                    heapq.heappush(cands, (-input_.rank, len(seen_edges), input_))
                    seen_edges.add((input_, cand))
    return seen_edges

class DotNode(object):
    def _shape(self):
        if isinstance(self.node, Variable):
            return "oval"
        elif isinstance(self.node, Split):
            return "hexagon"
        else:
            return "box"

    def _label(self):
        if isinstance(self.node, Variable):
            if self.node.data.shape == tuple():
                return str(self.node.data.dtype)
            return "%s, %s" % (str(self.node.data.shape), str(self.node.data.dtype))
        elif isinstance(self.node, chainer.functions.basic_math.Add):
            return "+"
        elif isinstance(self.node, chainer.functions.basic_math.AddConstant):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "+ %s" % str(value)
            elif isinstance(value, Variable):
                return "+ %s" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, chainer.functions.basic_math.Sub):
            return "-"
        elif isinstance(self.node, chainer.functions.basic_math.SubFromConstant):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "* (-1) + %s" % str(value)
            elif isinstance(value, Variable):
                return "* (-1) + %s" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, chainer.functions.basic_math.Mul):
            return "*"
        elif isinstance(self.node, chainer.functions.basic_math.MulConstant):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "* %s" % str(value)
            elif isinstance(value, Variable):
                return "* %s" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, chainer.functions.basic_math.Div):
            return "/"
        elif isinstance(self.node, chainer.functions.basic_math.DivFromConstant):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "/ %s" % str(value)
            elif isinstance(value, Variable):
                return "/ %s" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, chainer.functions.basic_math.PowVarVar):
            return "**"
        elif isinstance(self.node, chainer.functions.basic_math.PowVarConst):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "** %s" % str(value)
            elif isinstance(value, Variable):
                return "** %s" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, chainer.functions.basic_math.PowConstVar):
            value = self.node.value
            if isinstance(value, float) or isinstance(value, np.ndarray):
                return "%s **" % str(value)
            elif isinstance(value, Variable):
                return "%s **" % str(value.data)
            else:
                raise ValueError('value must be float, ndarray, or Variable')
        elif isinstance(self.node, chainer.functions.basic_math.Exp):
            return "exp"
        elif isinstance(self.node, chainer.functions.basic_math.Log):
            return "log"
        else:
            return str(type(self.node))

    def __init__(self, node):
        self.node = node
        self.id_ = id(node)
        self.attribute = {
            "label" : self._label(),
            "shape" : self._shape()
        }

    def __str__(self):
        return "%s [%s];" % (self.id_, ",".join(["%s=\"%s\""%(k, v) for (k, v) in self.attribute.items()]))

def generate_dot(edges):
    ret = "digraph graphname{"
    for edge in edges:
        head, tail = edge
        assert (isinstance(head, Variable) and isinstance(tail, Function)) or \
            (isinstance(head, Function) and isinstance(tail, Variable))
        head_node = DotNode(head)
        tail_node = DotNode(tail)
        ret += str(head_node)
        ret += str(tail_node)
        ret += "%s -> %s;" % (head_node.id_, tail_node.id_)
    ret += "}"
    return ret
            
def print_graph(outputs, remove_split=False):
    return generate_dot(build_graph(outputs, remove_split))
