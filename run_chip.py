import cupy_backends.cuda.api.runtime  # This needs to be imported first for some reason...
import cupy

cupy.show_config()
a = cupy.arange(10)
print(a)
print(a + 3)
print(a.sum())
