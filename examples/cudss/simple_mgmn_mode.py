# SPDX-FileCopyrightText: 2023-2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""CuPy port of NVIDIA's cuDSS/simple_mgmn_mode example (OpenMPI or NCCL).

https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuDSS/simple_mgmn_mode
"""
import argparse
import ctypes
import os

from _common import check_solution, system


def solve(args, MPI):
    import cupy as cp
    from cupy_backends.cuda.libs import cudss

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    # Both bundled cuDSS communication layers use OpenMPI for host buffers.
    if 'Open MPI' not in MPI.Get_library_version():
        raise RuntimeError('The bundled cuDSS layers require OpenMPI')

    # Use the node-local rank so ranks on different nodes can reuse GPU IDs.
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        local_rank, local_size = local.Get_rank(), local.Get_size()
    finally:
        local.Free()
    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count == 0:
        raise RuntimeError('No CUDA devices are visible')
    if device_count == 1:
        # Also supports launchers that expose one assigned GPU per process.
        device = 0
    elif local_size <= device_count:
        device = local_rank
    else:
        raise RuntimeError('Launch at most one MPI rank per GPU on each node')
    cp.cuda.Device(device).use()

    nccl_comm = None
    try:
        # cuDSS wants the address of a C communicator, not a Fortran handle.
        host_address = MPI._addressof(comm)
        host_size = MPI._sizeof(MPI.Comm)
        device_address, device_size = host_address, host_size
        if args.backend == 'nccl':
            from cupy_backends.cuda.libs import nccl

            unique_id = nccl.get_unique_id() if rank == 0 else None
            unique_id = comm.bcast(unique_id, root=0)
            nccl_comm = nccl.NcclCommunicator(size, unique_id, rank)
            device_comm = ctypes.c_void_p(nccl_comm.comm)
            device_address = ctypes.addressof(device_comm)
            device_size = ctypes.sizeof(device_comm)

        def configure_handle(handle):
            cudss.setCommLayer(handle, args.comm_layer)

        with system(root=(rank == 0),
                    configure_handle=configure_handle) as problem:
            cudss.dataSet(problem.handle, problem.data,
                          cudss.CUDSS_DATA_COMM_HOST, host_address, host_size)
            cudss.dataSet(problem.handle, problem.data,
                          cudss.CUDSS_DATA_COMM_DEVICE,
                          device_address, device_size)
            # Every rank participates; only rank 0 supplies matrix buffers.
            for phase in (cudss.CUDSS_PHASE_ANALYSIS,
                          cudss.CUDSS_PHASE_FACTORIZATION,
                          cudss.CUDSS_PHASE_SOLVE):
                cudss.execute(problem.handle, phase,
                              problem.config, problem.data,
                              problem.matrix, problem.x, problem.b)
            if rank == 0:
                check_solution(problem)
    finally:
        # Keep both communicators alive until all cuDSS objects are destroyed.
        if nccl_comm is not None:
            nccl_comm.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=['openmpi', 'nccl'])
    parser.add_argument('comm_layer', nargs='?',
                        help='library path; otherwise use CUDSS_COMM_LIB')
    args = parser.parse_args()
    if not (args.comm_layer or os.environ.get('CUDSS_COMM_LIB')):
        parser.error('provide a communication layer or set CUDSS_COMM_LIB')

    from mpi4py import MPI

    try:
        solve(args, MPI)
    except Exception:
        # A failed rank must not leave peers waiting in a cuDSS collective.
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
        raise


if __name__ == '__main__':
    main()
