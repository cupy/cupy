# from cupy.cuda cimport compiler
# from cupy.cuda cimport memory
# from cupy.cuda cimport module
# from cupy.cuda cimport stream

# compile_with_cache = compiler.compile_with_cache

from cupy.cuda.device cimport Device

# alloc = memory.alloc
# Memory = memory.Memory
# MemoryPointer = memory.MemoryPointer
# MemoryPool = memory.MemoryPool
# set_allocator = memory.set_allocator

from cupy.cuda.module cimport CPointer

# Event = stream.Event
# Stream = stream.Stream
# get_elapsed_time = stream.get_elapsed_time
