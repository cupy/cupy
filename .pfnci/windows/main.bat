:: Example: main.bat 3.7.0 9.0 master

set PYTHON=%1
set CUDA=%2

:: Set environment variables
call _use_cuda.bat %CUDA%
call _use_python.bat %PYTHON%

:: Show environment variables
set
python -V


:: Install dependencies
python -m pip install -U Cython
python -m pip list

:: Build
cd cupy
python -m pip install -e ".[jenkins]" -vvv

:: Test import
python -c "import cupy; cupy.show_config()"

:: Run unit tests
python -m pytest tests
