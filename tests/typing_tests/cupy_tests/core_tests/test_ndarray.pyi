import numpy
import cupy
import cupy as xp


stream = cupy.cuda.stream.get_current_stream()
x = xp.empty((10, 20, 30), dtype=float)
x_cp = cupy.empty((10, 20, 30), dtype=float)


x.view()
x.view(None)
x.view(float)

x.__cuda_array_interface__
x.__dlpack__()
x.__dlpack__(stream=stream)
x.__dlpack_device__()

is_contiguous: bool
flags = x.flags
is_contiguous = flags.c_contiguous
is_contiguous = flags.f_contiguous
is_contiguous = flags["C_CONTIGUOUS"]
is_contiguous = flags["F_CONTIGUOUS"]
owndata: bool = flags["OWNDATA"]

shape: tuple[int, ...] = x.shape
strides: tuple[int, ...] = x.strides
ndim: int = x.ndim
itemsize: int = x.itemsize
nbytes: int = x.nbytes

x = x.T
x.flat
# x_cp.cstruct
item_f: float = xp.empty((), dtype=float).item()
item_i: int = xp.empty((), dtype=int).item()
x.tolist()
x.tobytes()
x.tobytes("C")
x.tobytes("F")
x.tobytes("A")
x.tobytes("K")
x.dump("filename")
dumps: bytes = x.dumps()

x.astype(float)
x.astype(int)
x.astype("int64")
x.astype(float, order="C")
x.astype(float, order="F")
x.astype(float, order="A")
x.astype(float, order="K")
x.astype(float, "C", "unsafe", True, False)
x.astype(dtype=float, order="C", casting="unsafe", subok=True, copy=False)

x.copy()
x.copy(order="F")

x.fill(10)
x.fill(None)

x.reshape((10, 20, 30))
x.reshape([10, 20, 30])
x.reshape(10, 20, 30)
x.reshape(10, order="C")
x.reshape(10, order="F")
x.reshape(10, order="A")
x.reshape(10, order="K")  # E: No overload variant [call-overload]
x.reshape(10, "C")  # E: No overload variant [call-overload]
x.reshape()

x.transpose()
x.transpose(0, 1, 2)
x.transpose((0, 1, 2))
x.swapaxes(0, 1)
x.swapaxes(0, 1, 2)  # E: Too many arguments for [call-arg]

x.flatten()
x.flatten("C")
x.flatten("F")
x.flatten("A")
x.flatten("K")
x.ravel()
x.ravel("C")
x.ravel("F")
x.ravel("A")
x.ravel("K")

x.squeeze()
x.squeeze(0)
x.squeeze(axis=0)
