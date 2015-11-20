from cupy.cuda import compiler
from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import memory
from cupy.cuda import stream

compile_with_cache = compiler.compile_with_cache

Device = device.Device
get_cublas_handle = device.get_cublas_handle
get_device_id = device.get_device_id

alloc = memory.alloc
Memory = memory.Memory
MemoryPointer = memory.MemoryPointer
MemoryPool = memory.MemoryPool
set_allocator = memory.set_allocator

Function = function.Function
Module = function.Module

Event = stream.Event
Stream = stream.Stream
get_elapsed_time = stream.get_elapsed_time
