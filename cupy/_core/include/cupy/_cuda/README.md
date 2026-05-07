The `cuda-12` directory contains headers from CUDA Toolkit 12.1.1, redistributed under the following license:

https://docs.nvidia.com/cuda/archive/12.1.1/eula/#nvidia-cuda-toolkit-license-agreement

For CUDA 12.2+, we don't bundle header files as `cuda_fp16.h` started to depend
on header files that cannot be bundled (e.g. `vector_types.h`).
https://docs.cupy.dev/en/latest/install.html#cupy-always-raises-nvrtc-error-compilation-6
