import test_dtype


test_dtype.init()
print(type(test_dtype.complex32))
print(test_dtype.complex32)
print(test_dtype.complex32_dtype)


x = test_dtype.complex32_dtype
assert x.base.__repr__() == "dtype('complex32')"
assert x.byteorder == '='
assert x.char == 'E'
assert x.descr == [('', '<c4')]
assert x.fields is None
assert not x.hasobject
assert x.isnative
assert x.itemsize == 4
assert x.kind == 'c'
assert x.name == str(x) == 'complex32'
assert x.ndim == 0
assert x.str == '<c4'
assert str(x.type).endswith(".complex32'>")
