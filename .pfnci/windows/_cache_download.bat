if "%CUPY_CI_NO_CACHE%" == "1" (
    goto :eof
)

set ORIG_CD=%CD%
cd %USERPROFILE%
set CACHE_FILE=cupy_kernel_cache_windows.zip
call gsutil -m cp gs://tmp-asia-pfn-public-ci/cupy-ci/%CACHE_FILE% .
7z x %CACHE_FILE%
del %CACHE_FILE%
cd %ORIG_CD%
