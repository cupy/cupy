# Custom user structure examples

This folder contains examples of custom user structures in `cupy.RawKernel` (see [https://docs.cupy.dev/en/stable/tutorial/kernel.html](https://docs.cupy.dev/en/stable/tutorial/kernel.html) for corresponding documentation).

This folder provides three scripts ranked by increasing complexity:

1. `builtins_vectors.py` shows how to use Cuda builtin vectors such as `float4` both as kernel parameter and kernel arguments in RawKernels.
2. `packed_matrix.py` demonstrate how to create and use templated packed structures in RawModules.
3. `complex_struct.py` illustrates the possibility to recursively build complex numpy datatypes matching device structure memory layout.

All examples are run as simple python scripts: `python3.x example_name.py`.
