import cupy
import numpy

import cupy_perf


class Perf1(cupy_perf.PerfCases):
    enable_line_profiler = False

    def add_f(self, f):
        name = f.__name__
        setattr(self, name, f)
        setattr(self, name + '_fuse', cupy.fuse(f))
        setattr(self, name + '_fusex', cupy.fusex(f))

    def add_f2(self, f):
        name = f.__name__
        setattr(self, name, f)
        setattr(self, name + '_fusex', cupy.fusex(f))

    def setUp(self):
        shape_tiny = (2, 1, 2)
        shape_huge = (2000, 50, 100)
        self.a = cupy.empty(shape_tiny, numpy.float32)
        self.b = cupy.empty(shape_tiny, numpy.float32)
        self.c = cupy.empty(shape_tiny, numpy.float32)
        self.a_huge = cupy.empty(shape_huge, numpy.float32)
        self.b_huge = cupy.empty(shape_huge, numpy.float32)
        self.c_huge = cupy.empty(shape_huge, numpy.float32)

        def add20(x):
            return x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x

        self.add_f(add20)

        def sum(x):
            return cupy.sum(x)

        self.add_f(add20)
        self.add_f(sum)

        def sum10(x):
            return cupy.sum(x) + cupy.sum(x) + cupy.sum(x) + cupy.sum(x) + cupy.sum(x) + cupy.sum(x) + cupy.sum(x) + cupy.sum(x) + cupy.sum(x) + cupy.sum(x)

        self.add_f2(sum10)

    def perf_add20(self):
        self.add20(self.a)

    def perf_add20_fuse(self):
        self.add20_fuse(self.a)

    def perf_add20_fusex(self):
        self.add20_fusex(self.a)

    def perf_sum(self):
        return self.sum(self.a)

    def perf_sum_fuse(self):
        return self.sum_fuse(self.a)

    def perf_sum_fusex(self):
        return self.sum_fusex(self.a)

    def perf_sum10(self):
        return self.sum10(self.a)

    def perf_sum10_fusex(self):
        return self.sum10_fusex(self.a)

    # @cupy_perf.attr(n=500)
    # def perf_sum_huge(self):
    #     a = cupy.sum(self.a_huge)
    #     assert a.shape == ()

cupy_perf.run(__name__)
