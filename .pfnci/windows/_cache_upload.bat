if "%CUPY_CI_NO_CACHE%" == "1" (
    goto :eof
)

python .pfnci\trim_cupy_kernel_cache.py --expiry 259200 --rm

cd %USERPROFILE%
7z a -tzip cupy_kernel_cache.zip .cupy
gsutil -m cp cupy_kernel_cache_windows.zip gs://tmp-asia-pfn-public-ci/cupy-ci/
del cupy_kernel_cache_windows.zip
