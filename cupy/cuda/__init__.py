from cupy.cuda import compiler
from cupy.cuda import device
from cupy.cuda import memory
from cupy.cuda import module
from cupy.cuda import stream

compile_with_cache = compiler.compile_with_cache

Device = device.Device
clear_device_dependent_memo = device.clear_device_dependent_memo
memoize = device.memoize
using_device = device.using_device

alloc = memory.alloc
Memory = memory.Memory
MemoryPointer = memory.MemoryPointer
MemoryPool = memory.MemoryPool
set_default_allocator = memory.set_default_allocator

Function = module.Function
Module = module.Module

Event = stream.Event
Stream = stream.Stream
get_elapsed_time = stream.get_elapsed_time
