import cupy
import cupy as cp
import numpy
from cupy.core.fusionx import _FusionXHistory, FusionX

import cupy_perf

from batchnorm.gen_input import gen_input

class PerfTest(cupy_perf.PerfCases):
    enable_line_profiler = False

    def setUp(self):
        self.x = cp.arange(10101010)

        def f(x):
            return cp.sum(x) + cp.sum(x) + cp.sum(x)

        self.f = f
        self.fused = cp.fusex(f)

    def perf_nofuse(self):
        self.f(self.x)

    def perf_fuse(self):
        self.fused(self.x)

_FusionXHistory.exec = profile(_FusionXHistory.exec)
FusionX.__call__ = profile(FusionX.__call__)
# cupy_perf.run(__name__)

hoge = PerfTest()
hoge.setUp()
for i in range(100):
    hoge.perf_fuse()
