if "%CUPY_CI_NO_CACHE%" == "1" (
    goto :eof
)

python .pfnci\trim_cupy_kernel_cache.py --expiry 259200 --rm

set ORIG_CD=%CD%
cd %USERPROFILE%
set CACHE_FILE=cupy_kernel_cache_windows.zip

:: -mx=0 ... no compression
:: -mtc=on ... preserve timestamp
7z a -tzip -mx=0 -mtc=on %CACHE_FILE% .cupy
call gsutil -m cp %CACHE_FILE% gs://tmp-asia-pfn-public-ci/cupy-ci/
del %CACHE_FILE%
cd %ORIG_CD%
