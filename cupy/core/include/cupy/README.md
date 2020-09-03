# "include" directory

All files and directories in this directory will be copied to the distribution (sdist and wheel).
Note that items starting with `.` (e.g., `cub/.git`) are excluded.
See `setup.py` for details.

## CUB

The `cub` folder is a git submodule for the CUB project.
Including the CUB headers as a submodule enables not only building the `cupy.cuda.cub` module,
but also easier maintenance.
For further information on CUB, see the [CUB Project Website](http://nvlabs.github.com/cub).
