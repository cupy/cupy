import cupy


class TestClassGetItem:

    def test_class_getitem(self):
        from typing import Any
        cupy.ndarray[Any, Any]
