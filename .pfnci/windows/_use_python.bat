:: This file must be invoked via "call" from another batch.

set VERSION=%1

if %VERSION% == 3.6 (
    set PYTHON_ROOT=C:\Development\Python\Python36
) else if %VERSION% == 3.7 (
    set PYTHON_ROOT=C:\Development\Python\Python37
) else if %VERSION% == 3.8 (
    set PYTHON_ROOT=C:\Development\Python\Python38
) else (
    echo Unsupported Python version: %VERSION%
    exit /b 1
)

set PATH=%PYTHON_ROOT%;%PYTHON_ROOT%\Scripts;%PATH%

:: Unset
set VERSION=
set PYTHON_ROOT=
