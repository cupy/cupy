# nvprof --print-gpu-trace python examples/stream/curand.py
import cupy

x = cupy.array([1, 2, 3])
rand = cupy.random.generator.RandomState()

with cupy.cuda.stream.Stream():
    y = rand.lognormal(size=(1, 3))

stream = cupy.cuda.stream.Stream()
stream.use()
y = rand.lognormal(size=(1, 3))
