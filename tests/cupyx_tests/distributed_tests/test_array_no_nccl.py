import importlib
import os
import sys

import pytest

import cupyx.distributed._array


@pytest.fixture(autouse=True)
def make_nccl_unavailable(monkeypatch):
    class MockNcclModule:
        available = False

    monkeypatch.setattr(cupy.cuda, 'nccl', MockNcclModule())

    importlib.reload(cupyx.distributed._array)


sys.path.append(os.getcwd())
from test_array_nccl import *
sys.path.pop()
