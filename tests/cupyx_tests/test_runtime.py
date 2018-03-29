import unittest

import cupy
import cupyx


class TestRuntime(unittest.TestCase):
    def test_runtime(self):
        runtime = cupyx.get_runtime_info()
        assert cupy.__version__ == runtime.cupy_version
        assert cupy.__version__ in str(runtime.cupy_version)
