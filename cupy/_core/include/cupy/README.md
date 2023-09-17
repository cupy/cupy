# "include" directory

All files and directories in this directory will be copied to the distribution (sdist and wheel).
Note that items starting with `.` (e.g., `cub/.git`) are excluded.
See `setup.py` for details.

## CCCL

The `cccl` folder is a git submodule for the CCCL project, the new home for Thrust, CUB, and
libcu++. The CCCL headers are used at both build- and run- time.
For further information on CUB, see the [CCCL repo](https://github.com/NVIDIA/cccl).

## Jitify
The `Jitify` folder is a git submodule for the Jitify project.
Including the Jitify header as a submodule for building the `cupy.cuda.jitify` module.
For further information on Jitify, see the [Jitify repo](https://github.com/NVIDIA/jitify).

## DLPack
The `dlpack` folder stores the DLPack header for building the `cupy._core.dlpack` module,
see `README.md` therein.
For further information on DLPack, see the [DLPack repo](https://github.com/dmlc/dlpack).
