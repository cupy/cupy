:: Example: main.bat 10.0 3.7

set CUDA=%1
set PYTHON=%2
set TARGET=%3

:: Set environment variables
call .pfnci\windows\_use_cuda.bat %CUDA% || goto :error
call .pfnci\windows\_use_python.bat %PYTHON% || goto :error

:: Show environment variables
set || goto :error
python -V || goto :error

:: Install dependencies
python -m pip install -U Cython scipy optuna || goto :error
python -m pip list || goto :error

:: Build
set CUPY_NUM_BUILD_JOBS=16
set CUPY_NVCC_GENERATE_CODE=current
python -m pip install -e ".[jenkins]" -vvv || goto :error

:: Test import
python -c "import cupy; cupy.show_config()" || goto :error

:: Exit if build only mode
if "%TARGET%" == "build" (
  goto :eof
)

:: Run unit tests
set PYTEST_OPTS=-m "not slow"
if "%TARGET%" == "slow" (
  set PYTEST_OPTS=-m "slow"
)
call .pfnci\windows\_cache_download.bat
python -m pytest -rfEX %PYTEST_OPTS% tests || goto :error
call .pfnci\windows\_cache_upload.bat


goto :EOF

:: Error handling
:error
echo Failed with status %errorlevel%.
exit /b %errorlevel%
