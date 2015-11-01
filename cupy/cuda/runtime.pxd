###############################################################################
# Types
###############################################################################

from cupy.cuda.driver cimport Device
from cupy.cuda.driver cimport Event
from cupy.cuda.driver cimport Stream
from cupy.cuda.driver cimport StreamCallback

cdef struct cudaPointerAttributes:
    int device
    void* devicePointer
    void* hostPointer
    int isManaged
    int memoryType


cpdef int getDevice()
cpdef setDevice(int device)

