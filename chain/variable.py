import heapq

class Variable(object):
    """Variable node."""

    def __init__(self, data, rank=0):
        self.data = data
        self.rank = rank

        self.grad = None
        self.creator = None

    def __pos__(self):
        return self

    def set_creator(self, gen_func):
        """Set function that creates this variable."""

        self.creator = gen_func
        self.rank = gen_func.rank + 1

    def backward(self):
        """Run error backpropagation from this variable node."""

        cand_funcs = []
        seen_set = set()

        def add_cand(cand):
            if cand is not None and cand not in seen_set:
                # Negate since heapq is min-heap
                heapq.heappush(cand_funcs, (-cand.rank, cand))
                seen_set.add(cand)

        add_cand(self.creator)

        while cand_funcs:
            _, func = heapq.heappop(cand_funcs)
            gxs = func.backward(tuple(x.data for x in func.inputs),
                                tuple(y.grad for y in func.outputs))
            for x, gx in zip(func.inputs, gxs):
                x.grad = gx
                if gx is not None:  # skip if gradient does not flow
                    add_cand(x.creator)

    def forget_backward(self):
        """Delete nodes backward through the graph."""

        cand_funcs = []
        seen_set = set()

        def add_cand(cand):
            if cand is not None and cand not in seen_set:
                cand_funcs.append(cand)
                seen_set.add(cand)

        add_cand(self.creator)

        while cand_funcs:
            func = cand_funcs.pop()
            for var in func.inputs:
                add_cand(var.creator)
            func.forget()
