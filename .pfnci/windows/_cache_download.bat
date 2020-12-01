if "%CUPY_CI_NO_CACHE%" == "1" (
    goto :eof
)

set ORIG_CD=%CD%
cd %USERPROFILE%
call gsutil -m cp gs://tmp-asia-pfn-public-ci/cupy-ci/cupy_kernel_cache_windows.zip .
7z x cupy_kernel_cache_windows.zip
del cupy_kernel_cache_windows.zip
cd %ORIG_CD%
