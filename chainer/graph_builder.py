import heapq
from chainer import Variable, Function
from chainer.function import Split

def build_graph(outputs):
    cands = []
    seen_edges = set()

    for o in outputs:
        heapq.heappush(cands, (-o.rank, len(seen_edges), o))

    while cands:
        _, _, cand = heapq.heappop(cands)
        if isinstance(cand, Variable):
            creator = cand.creator
            if creator is not None and (creator, cand) not in seen_edges:
                heapq.heappush(cands, (-creator.rank, len(seen_edges), creator))
                seen_edges.add((creator, cand))
        elif isinstance(cand, Function):
            for i in cand.inputs:
                if i != cand and (i, cand) not in seen_edges:
                    heapq.heappush(cands, (-i.rank, len(seen_edges), i))
                    seen_edges.add((i, cand))
    return seen_edges

def generate_dot(edges):
    ret = "digraph graphname{"
    for edge in edges:
        head, tail = edge
        assert (isinstance(head, Variable) and isinstance(tail, Function)) or \
            (isinstance(head, Function) and isinstance(tail, Variable))
        if isinstance(head, Variable):
            ret += "v%d [shape=oval, label=\"%s\"];" % (id(head), str(head.data.shape))
            if isinstance(tail, Split):
                ret += "s%d [shape=hexagon];" % id(tail)
                ret += "v%d -> s%d;" % (id(head), id(tail))
            else:
                ret += "f%d [shape=box];" % id(tail)
                ret += "v%d -> f%d;" % (id(head), id(tail))
        else:
            ret += "v%d [shape=oval, label=\"%s\"];" % (id(tail), str(tail.data.shape))
            if isinstance(head, Split):
                ret += "s%d [shape=hexagon];" % (id(head))
                ret += "s%d -> v%d;" % (id(head), id(tail))
            else:
                ret += "f%d [shape=box];" % (id(head))
                ret += "f%d -> v%d;" % (id(head), id(tail))

    ret += "}"
    return ret
            
def print_graph(outputs):
    return generate_dot(build_graph(outputs))
