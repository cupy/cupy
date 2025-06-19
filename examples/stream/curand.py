# nvprof --print-gpu-trace python examples/stream/curand.py
import cupy

rand = cupy.random.RandomState()

stream = cupy.cuda.stream.Stream()
with stream:
    y = rand.lognormal(size=(1, 3))

stream = cupy.cuda.stream.Stream()
stream.use()
y = rand.lognormal(size=(1, 3))
