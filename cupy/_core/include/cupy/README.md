# "include" directory

All files and directories in this directory will be copied to the distribution (sdist and wheel).
Note that items starting with `.` (e.g., `cub/.git`) are excluded.
See `setup.py` for details.

## CUB

The `cub` folder is a git submodule for the CUB project.
Including the CUB headers as a submodule enables not only building the `cupy.cuda.cub` module,
but also easier maintenance.
For further information on CUB, see the [CUB Project Website](http://nvlabs.github.com/cub).

## Jitify
The `Jitify` folder is a git submodule for the Jitify project.
Including the Jitify header as a submodule for building the `cupy.cuda.jitify` module.
For further information on Jitify, see the [Jitify repo](https://github.com/NVIDIA/jitify).

## DLPack
The `dlpack` folder stores the DLPack header for building the `cupy._core.dlpack` module,
see `README.md` therein.
For further information on DLPack, see the [DLPack repo](https://github.com/dmlc/dlpack).
