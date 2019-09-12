import cupy
import numpy

import cupy_perf


class Perf1(cupy_perf.PerfCases):
    enable_line_profiler = False

    def setUp(self):
        shape_tiny = (2, 1, 2)
        shape_huge = (2000, 50, 100)
        self.a = cupy.empty(shape_tiny, numpy.float32)
        self.b = cupy.empty(shape_tiny, numpy.float32)
        self.c = cupy.empty(shape_tiny, numpy.float32)
        self.a_huge = cupy.empty(shape_huge, numpy.float32)
        self.b_huge = cupy.empty(shape_huge, numpy.float32)
        self.c_huge = cupy.empty(shape_huge, numpy.float32)

        self.kernel = self._get_kernel()

    def _get_kernel(self):
        return cupy.ElementwiseKernel(
            '''T a, T b''', '''T c''',
            '''c = a + b;''', 'test_kernel')

    def perf_empty(self):
        pass

    def perf_sum(self):
        cupy.sum(self.a)

    @cupy_perf.attr(n=500)
    def perf_sum_huge(self):
        a = cupy.sum(self.a_huge)
        assert a.shape == ()

    def perf_add(self):
        cupy.add(self.a, self.b)

    def perf_add_out(self):
        cupy.add(self.a, self.b, out=self.c)

    def perf_add_out_huge(self):
        cupy.add(self.a_huge, self.b_huge, out=self.c_huge)

    def perf_userkernel_create(self):
        self._get_kernel()

    def perf_userkenel(self):
        self.kernel(self.a, self.b, self.c)

    @cupy_perf.attr(n=10000)
    def perf_userkernel_huge(self):
        self.kernel(self.a_huge, self.b_huge, self.c_huge)


cupy_perf.run(__name__)
