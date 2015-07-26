from cupy.cuda import compiler
from cupy.cuda import device
from cupy.cuda import memory
from cupy.cuda import module
from cupy.cuda import stream

compile_with_cache = compiler.compile_with_cache

Device = device.Device
DeviceUser = device.DeviceUser
memoize = device.memoize

alloc = memory.alloc
Memory = memory.Memory
MemoryPointer = memory.MemoryPointer
MemoryPool = memory.MemoryPool
set_default_allocator = memory.set_default_allocator

Function = module.Function
Module = module.Module

Event = stream.Event
Stream = stream.Stream
