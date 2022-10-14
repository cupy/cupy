from typing import Mapping, Optional, Sequence, Union, TYPE_CHECKING

import numpy
import numpy.typing as npt

import cupy
from cupy._core._scalar import get_typename

if TYPE_CHECKING:
    from cupyx.jit._internal_types import Data


# Base class for cuda types.
class TypeBase:

    def __str__(self) -> str:
        raise NotImplementedError

    def declvar(self, x: str, init: Optional['Data']) -> str:
        if init is None:
            return f'{self} {x}'
        return f'{self} {x} = {init.code}'

    def assign(self, var: 'Data', value: 'Data') -> str:
        return f'{var.code} = {value.code}'


class Void(TypeBase):

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return 'void'


class Scalar(TypeBase):

    def __init__(self, dtype: npt.DTypeLike) -> None:
        self.dtype = numpy.dtype(dtype)

    def __str__(self) -> str:
        dtype = self.dtype
        if dtype == numpy.float16:
            # For the performance
            dtype = numpy.dtype('float32')
        return get_typename(dtype)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, TypeBase)
        return isinstance(other, Scalar) and self.dtype == other.dtype

    def __hash__(self) -> int:
        return hash(self.dtype)


class PtrDiff(Scalar):

    def __init__(self) -> None:
        super().__init__('q')

    def __str__(self) -> str:
        return 'ptrdiff_t'


class ArrayBase(TypeBase):

    def __init__(self, child_type: TypeBase, ndim: int) -> None:
        assert isinstance(child_type, TypeBase)
        self.child_type = child_type
        self.ndim = ndim


class CArray(ArrayBase):

    def __init__(
            self,
            dtype: npt.DTypeLike,
            ndim: int,
            is_c_contiguous: bool,
            index_32_bits: bool,
    ) -> None:
        self.dtype = numpy.dtype(dtype)
        self._c_contiguous = is_c_contiguous
        self._index_32_bits = index_32_bits
        super().__init__(Scalar(dtype), ndim)

    @classmethod
    def from_ndarray(cls, x: cupy.ndarray) -> 'CArray':
        return CArray(x.dtype, x.ndim, x._c_contiguous, x._index_32_bits)

    def __str__(self) -> str:
        ctype = get_typename(self.dtype)
        c_contiguous = get_cuda_code_from_constant(self._c_contiguous, bool_)
        index_32_bits = get_cuda_code_from_constant(self._index_32_bits, bool_)
        return f'CArray<{ctype}, {self.ndim}, {c_contiguous}, {index_32_bits}>'

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, TypeBase)
        return (
            isinstance(other, CArray) and
            self.dtype == other.dtype and
            self.ndim == other.ndim and
            self._c_contiguous == other._c_contiguous and
            self._index_32_bits == other._index_32_bits
        )

    def __hash__(self) -> int:
        return hash(
            (self.dtype, self.ndim, self._c_contiguous, self._index_32_bits))


class SharedMem(ArrayBase):

    def __init__(
            self,
            child_type: TypeBase,
            size: Optional[int],
            alignment: Optional[int] = None,
    ) -> None:
        if not (isinstance(size, int) or size is None):
            raise 'size of shared_memory must be integer or `None`'
        if not (isinstance(alignment, int) or alignment is None):
            raise 'alignment must be integer or `None`'
        self._size = size
        self._alignment = alignment
        super().__init__(child_type, 1)

    def declvar(self, x: str, init: Optional['Data']) -> str:
        assert init is None
        if self._alignment is not None:
            code = f'__align__({self._alignment})'
        else:
            code = ''
        if self._size is None:
            code = f'extern {code} __shared__ {self.child_type} {x}[]'
        else:
            code = f'{code} __shared__ {self.child_type} {x}[{self._size}]'
        return code


class Ptr(ArrayBase):

    def __init__(self, child_type: TypeBase) -> None:
        super().__init__(child_type, 1)

    def __str__(self) -> str:
        return f'{self.child_type}*'


class Tuple(TypeBase):

    def __init__(self, types: Sequence[TypeBase]) -> None:
        self.types = types

    def __str__(self) -> str:
        types = ', '.join([str(t) for t in self.types])
        return f'thrust::tuple<{types}>'

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, TypeBase)
        return isinstance(other, Tuple) and self.types == other.types


void: Void = Void()
bool_: Scalar = Scalar(numpy.bool_)
int32: Scalar = Scalar(numpy.int32)
uint32: Scalar = Scalar(numpy.uint32)
uint64: Scalar = Scalar(numpy.uint64)


class Dim3(TypeBase):
    """
    An integer vector type based on uint3 that is used to specify dimensions.

    Attributes:
        x (uint32)
        y (uint32)
        z (uint32)
    """

    def x(self, code: str) -> 'Data':
        from cupyx.jit import _internal_types  # avoid circular import
        return _internal_types.Data(f'{code}.x', uint32)

    def y(self, code: str) -> 'Data':
        from cupyx.jit import _internal_types  # avoid circular import
        return _internal_types.Data(f'{code}.y', uint32)

    def z(self, code: str) -> 'Data':
        from cupyx.jit import _internal_types  # avoid circular import
        return _internal_types.Data(f'{code}.z', uint32)

    def __str__(self) -> str:
        return 'dim3'


dim3: Dim3 = Dim3()


_suffix_literals_dict: Mapping[str, str] = {
    'float64': '',
    'float32': 'f',
    'int64': 'll',
    'int32': '',
    'uint64': 'ull',
    'uint32': 'u',
    'bool': '',
}


def get_cuda_code_from_constant(
        x: Union[bool, int, float, complex],
        ctype: Scalar,
) -> str:
    dtype = ctype.dtype
    suffix_literal = _suffix_literals_dict.get(dtype.name)
    if suffix_literal is not None:
        s = str(x).lower()
        return f'{s}{suffix_literal}'
    ctype_str = str(ctype)
    if dtype.kind == 'c':
        return f'{ctype_str}({x.real}, {x.imag})'
    if ' ' in ctype_str:
        return f'({ctype_str}){x}'
    return f'{ctype_str}({x})'
