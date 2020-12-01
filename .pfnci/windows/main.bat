:: Example: main.bat 10.0 3.7

set CUDA=%1
set PYTHON=%2

:: Set environment variables
call .pfnci\windows\_use_cuda.bat %CUDA% || goto :error
call .pfnci\windows\_use_python.bat %PYTHON% || goto :error

:: Show environment variables
set || goto :error
python -V || goto :error

:: Install dependencies
python -m pip install -U Cython || goto :error
python -m pip list || goto :error

:: Build
set CUPY_NUM_BUILD_JOBS=16
set CUPY_NVCC_GENERATE_CODE=current
python -m pip install -e ".[jenkins]" -vvv || goto :error

:: Test import
python -c "import cupy; cupy.show_config()" || goto :error

:: Run unit tests
call .pfnci\windows\_cache_download.bat
python -m pytest tests || goto :error
call .pfnci\windows\_cache_upload.bat


goto :EOF

:: Error handling
:error
echo Failed with status %errorlevel%.
exit /b %errorlevel%
