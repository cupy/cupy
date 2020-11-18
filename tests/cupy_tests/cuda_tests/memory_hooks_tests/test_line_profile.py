import io
import unittest
import re

from cupy.cuda import memory
from cupy.cuda import memory_hooks
from cupy import testing


@testing.gpu
class TestLineProfileHook(unittest.TestCase):

    def setUp(self):
        self.pool = memory.MemoryPool()

    def test_print_report(self):
        hook = memory_hooks.LineProfileHook()
        p = self.pool.malloc(1000)
        del p
        with hook:
            p1 = self.pool.malloc(1000)
            p2 = self.pool.malloc(2000)
        del p1
        del p2
        f = io.StringIO()
        hook.print_report(file=f)
        actual = f.getvalue()
        expect = r'\A_root \(3\.00KB, 2\.00KB\)'
        assert re.search(expect, actual)
        expect = r'.*\.py:[0-9]+:test_print_report \(1\.00KB, 0\.00B\)'
        assert re.search(expect, actual)
        expect = r'.*\.py:[0-9]+:test_print_report \(2\.00KB, 2\.00KB\)'
        assert re.search(expect, actual)

    def test_print_report_max_depth(self):
        hook = memory_hooks.LineProfileHook(max_depth=1)
        with hook:
            p = self.pool.malloc(1000)
        del p
        f = io.StringIO()
        hook.print_report(file=f)
        actual = f.getvalue()
        assert 2 == len(actual.split('\n'))

        hook = memory_hooks.LineProfileHook(max_depth=2)
        with hook:
            p = self.pool.malloc(1000)
        del p
        f = io.StringIO()
        hook.print_report(file=f)
        actual = f.getvalue()
        assert 3 == len(actual.split('\n'))
