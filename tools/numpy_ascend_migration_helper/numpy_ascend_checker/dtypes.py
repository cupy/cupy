"""dtype canonicalization.

All dtype spellings are canonicalized to one of the canonical names stored in
the rule DB (design doc §11):

    np.float32 / "float32" / "f4" / "<f4"  ->  "float32"
"""

from __future__ import annotations

from typing import Optional

ALIASES = {
    "bool": "bool",
    "bool_": "bool",
    "b1": "bool",
    "?": "bool",
    "int8": "int8",
    "i1": "int8",
    "int16": "int16",
    "i2": "int16",
    "int32": "int32",
    "i4": "int32",
    "int64": "int64",
    "i8": "int64",
    "uint8": "uint8",
    "u1": "uint8",
    "uint16": "uint16",
    "u2": "uint16",
    "uint32": "uint32",
    "u4": "uint32",
    "uint64": "uint64",
    "u8": "uint64",
    "float16": "float16",
    "f2": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "f4": "float32",
    "float64": "float64",
    "f8": "float64",
    "complex64": "complex64",
    "c8": "complex64",
    "complex128": "complex128",
    "c16": "complex128",
    # numpy default dtypes
    "int": "int64",
    "i": "int64",
    "uint": "uint64",
    "u": "uint64",
    "float": "float64",
    "f": "float64",
    "complex": "complex128",
    "c": "complex128",
    # non-device dtypes
    "object": "object",
    "O": "object",
    "datetime64": "datetime64",
    "M8": "datetime64",
    "M": "datetime64",
    "timedelta64": "timedelta64",
    "m8": "timedelta64",
    "m": "timedelta64",
    "str_": "str_",
    "U": "str_",
    "unicode_": "str_",
    "bytes_": "bytes_",
    "S": "bytes_",
}

# Extra spellings valid as `np.<attr>` but not as string codes.
ATTR_EXTRA = {
    "int_": "int64",
    "float_": "float64",
    "complex_": "complex128",
}

_BYTEORDER_PREFIXES = "<>=|@"


def canonical_dtype(value) -> Optional[str]:
    """Canonicalize a dtype spelled as a string code or attribute name."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if v in ALIASES:
        return ALIASES[v]
    if v in ATTR_EXTRA:
        return ATTR_EXTRA[v]
    if len(v) >= 2 and v[0] in _BYTEORDER_PREFIXES:
        stripped = v[1:]
        if stripped in ALIASES:
            return ALIASES[stripped]
    return None


def is_dtype_attr(name: str) -> bool:
    return canonical_dtype(name) is not None
