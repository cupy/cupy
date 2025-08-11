from __future__ import annotations

import cupy as cp

import pytest

# "example string" or
# ("example string", "xfail message")
examples = [
    "uint8(1) + 2",
    "array([1], uint8) + int64(1)",
    "array([1], uint8) + array(1, int64)",
    "array([1.], float32) + float64(1.)",
    "array([1.], float32) + array(1., float64)",
    "array([1], uint8) + 1",
    "array([1], uint8) + 200",
    "array([100], uint8) + 200",
    "array([1], uint8) + 300",
    "uint8(1) + 300",
    "uint8(100) + 200",
    "float32(1) + 3e100",
    "array([1.0], float32) + 1e-14 == 1.0",
    "array([0.1], float32) == float64(0.1)",
    "array(1.0, float32) + 1e-14 == 1.0",
    "array([1.], float32) + 3",
    "array([1.], float32) + int64(3)",
    "3j + array(3, complex64)",
    "float32(1) + 1j",
    "int32(1) + 5j",
    # additional examples from the NEP text
    "int16(2) + 2",
    "int16(4) + 4j",
    "float32(5) + 5j",
    "bool_(True) + 1",
    "True + uint8(2)",
    # not in the NEP
    "1.0 + array([1, 2, 3], int8)",
    "array([1], float32) + 1j",
]


@pytest.mark.parametrize('example', examples)
@cp.testing.numpy_cupy_allclose(atol=1e-15, accept_error=OverflowError)
def test_nep50_examples(xp, example):
    dct = {'array': xp.array, 'uint8': xp.uint8, 'int64': xp.int64,
           'float32': xp.float32, 'float64': xp.float64, 'int16': xp.int16,
           'bool_': xp.bool_, 'int32': xp.int32, 'complex64': xp.complex64,
           'int8': xp.int8, }

    if isinstance(example, tuple):
        example, mesg = example
        pytest.xfail(mesg)

    result = eval(example, dct)
    return result
