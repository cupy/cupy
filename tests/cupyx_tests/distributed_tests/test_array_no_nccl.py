import os
import sys

import cupy.cuda


class MockUnavailableModule:
    available = False


cupy.cuda.nccl = MockUnavailableModule()


sys.path.append(os.getcwd())
from test_array_nccl import *   # NOQA
sys.path.pop()
