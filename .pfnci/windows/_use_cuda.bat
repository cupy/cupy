:: This file must be invoked via "call" from another batch.

set VERSION=%1

if %VERSION% == 8.0 (
    set CUDA_PATH=%CUDA_PATH_V8_0%
) else if %VERSION% == 9.0 (
    set CUDA_PATH=%CUDA_PATH_V9_0%
) else if %VERSION% == 9.1 (
    set CUDA_PATH=%CUDA_PATH_V9_1%
) else if %VERSION% == 9.2 (
    set CUDA_PATH=%CUDA_PATH_V9_2%
) else if %VERSION% == 10.0 (
    set CUDA_PATH=%CUDA_PATH_V10_0%
) else if %VERSION% == 10.1 (
    set CUDA_PATH=%CUDA_PATH_V10_1%
) else if %VERSION% == 10.2 (
    set CUDA_PATH=%CUDA_PATH_V10_2%
) else if %VERSION% == 11.0 (
    set CUDA_PATH=%CUDA_PATH_V11_0%
) else if %VERSION% == 11.1 (
    set CUDA_PATH=%CUDA_PATH_V11_1%
) else (
    echo Unsupported CUDA version: %VERSION%
    exit /b 1
)

set PATH=%CUDA_PATH%\bin;%ProgramFiles%\NVIDIA Corporation\NvToolsExt\bin\x64;%PATH%

:: Unset
set VERSION=
