# cuDSS examples with CuPy

Python/CuPy adaptations of NVIDIA's
[cuDSS samples](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuDSS),
using the CuPy backend bindings for cuDSS 0.8.0.

All four examples solve the same symmetric 5 × 5 system. Only its upper
triangle is stored in CSR format; the expected solution is `[1, 2, 3, 4, 5]`.
They check the solution and print `Example PASSED` on success.

| Script | NVIDIA source | Demonstrates |
| --- | --- | --- |
| `simple.py` | [simple](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuDSS/simple) | Analysis, factorization, solve |
| `simple_get_set.py` | [simple_get_set](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuDSS/simple_get_set) | Configuration, permutations, memory estimates, factorization information, diagonal/scaling arrays, iterative refinement |
| `simple_mgmn_mode.py` | [simple_mgmn_mode](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuDSS/simple_mgmn_mode) | MPI processes with OpenMPI or NCCL communication |
| `simple_multithreaded_mode.py` | [simple_multithreaded_mode](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuDSS/simple_multithreaded_mode) | CPU threading through the cuDSS threading layer |

## Requirements

- A working CUDA GPU and CuPy built with `cupy_backends.cuda.libs.cudss`.
- cuDSS 0.8.0 or later available to the dynamic loader.
- NumPy, used for host configuration/metadata buffers. Matrix and solution
  buffers are CuPy arrays on the GPU.

`_common.py` holds the shared matrix setup and resource cleanup. cuDSS takes
integer addresses (`array.data.ptr` for CuPy and `array.ctypes.data` for NumPy).
The arrays remain alive while cuDSS uses them, and the stream is synchronized
before descriptors and solver resources are released, including on errors.

Run the following commands from the repository root.

## Basic solve and get/set

```sh
python examples/cudss/simple.py
python examples/cudss/simple_get_set.py
python examples/cudss/simple_get_set.py --dtype float32 --index-dtype int64
```

The get/set example uses `CUDSS_MTYPE_SYMMETRIC` to retrieve inertia and pivot
information, following the original sample. The other examples use
`CUDSS_MTYPE_SPD`. The cuDSS 0.8.0 CSR API receives both `offsetType` and
`indexType`; `rowEnd` is passed as NULL (`0`).

## Multi-threaded mode

Install the cuDSS threading layer and its OpenMP runtime. Supply its full
library path or a library name resolvable by the dynamic loader:

```sh
python examples/cudss/simple_multithreaded_mode.py \
    /path/to/libcudss_mtlayer_gomp.so --threads 4
```

Alternatively:

```sh
export CUDSS_THREADING_LIB=/path/to/libcudss_mtlayer_gomp.so
python examples/cudss/simple_multithreaded_mode.py --threads 4
```

`--threads` sets `CUDSS_CONFIG_HOST_NTHREADS`. This is cuDSS CPU parallelism,
controlled by its threading layer.

## Multi-GPU multi-node (MGMN) mode

Additional requirements:

- `mpi4py` linked against OpenMPI, with the same OpenMPI ABI as the cuDSS
  communication layer. The bundled layers use OpenMPI for host communication.
- A matching cuDSS communication layer library.
- For the OpenMPI backend, a CUDA-aware OpenMPI installation.
- For the NCCL backend, NCCL and CuPy's NCCL bindings.

```sh
mpiexec -n 2 python examples/cudss/simple_mgmn_mode.py openmpi \
    /path/to/libcudss_commlayer_openmpi.so

mpiexec -n 2 python examples/cudss/simple_mgmn_mode.py nccl \
    /path/to/libcudss_commlayer_nccl.so
```

You can omit the library argument when `CUDSS_COMM_LIB` names the matching
layer. Use the MPI launcher's host/hostfile options to run across nodes.
Allocate one MPI rank per GPU on each node. Device selection uses the
node-local rank; launchers may instead expose one assigned GPU per process
through `CUDA_VISIBLE_DEVICES`.

All ranks create descriptors with the same global dimensions and participate
in each solver phase. Only rank 0 allocates the input/output buffers and
checks the solution, as in the original sample. Both `CUDSS_DATA_COMM_HOST`
and `CUDSS_DATA_COMM_DEVICE` are set for the cuDSS 0.8.0 interface. With NCCL,
MPI broadcasts the NCCL unique ID and remains the host communicator.

See NVIDIA's [advanced features documentation](https://docs.nvidia.com/cuda/cudss/0.8.0/advanced_features.html)
for the communication and threading layer requirements.

## License

These examples adapt the Apache-2.0 licensed NVIDIA samples to Python/CuPy.
The original NVIDIA copyright notices are retained in the Python files.
See [LICENSE](LICENSE) for the Apache License, Version 2.0.
