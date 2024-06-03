# Run this script with the following command:
#
#   mpiexec -n 2 python multple_devices.py
#
# This script executes simple communication and computation with 2 MPI
# processes, each of which uses a different GPU

import cupy
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size != 2:
    raise RuntimeError("run this script with 2 processes: mpiexec -n 2 ...")
device_count = cupy.cuda.runtime.getDeviceCount()
if device_count < 2:
    raise RuntimeError("this script requires 2 GPUs")

# Select device based on local MPI rank.
# Caveat: for simplicity we assume local_rank == rank here, which may or may
# not be the case depending how MPI processes are launched and how your code
# is written. For more robust usage, you may need to consult the user manual
# for your MPI library. For example:
# local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"))  # Open MPI
# local_rank = int(os.getenv("MV2_COMM_WORLD_LOCAL_RANK"))   # MVAPICH2
local_rank = rank
cupy.cuda.Device(local_rank).use()

# send-recv
if rank == 0:
    arr = cupy.empty(100, dtype=cupy.int64)
    comm.Recv(arr, source=1, tag=87)
    assert (arr == cupy.arange(100).astype(cupy.int64)).all()
else:
    arr = cupy.arange(100).astype(cupy.int64)
    comm.Send(arr, dest=0, tag=87)

# allreduce
arr1 = cupy.empty(1000)
arr2 = cupy.random.random(1000)
arr_total = arr2.copy()
comm.Allreduce(MPI.IN_PLACE, arr_total)  # in-place reduction
if rank == 0:
    comm.Recv(arr1, source=1, tag=88)
    comm.Send(arr2, dest=1, tag=89)
else:
    comm.Send(arr2, dest=0, tag=88)
    comm.Recv(arr1, source=0, tag=89)
assert (arr1 + arr2 == arr_total).all()

print("process {}: finished".format(rank))
