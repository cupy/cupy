Param(
    [String]$cuda,
    [String]$python,
    [String]$test
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

. "$PSScriptRoot\_flexci.ps1"

$cupy_kernel_cache_file = "cupy_kernel_cache_windows.zip"


function DownloadCache {
    pushd $Env:USERPROFILE
    gsutil -m cp gs://tmp-asia-pfn-public-ci/cupy-ci/$cupy_kernel_cache_file .
    if (-not $?) {
        echo "*** Kernel cache unavailable"
    } else {
        RunOrDie 7z x $cupy_kernel_cache_file
        rm $cupy_kernel_cache_file
    }
    popd
}

function UploadCache {
    RunOrDie python .pfnci\trim_cupy_kernel_cache.py --expiry 259200 --rm

    pushd $Env:USERPROFILE
    # -mx=0 ... no compression
    # -mtc=on ... preserve timestamp
    RunOrDie 7z a -tzip -mx=0 -mtc=on $cupy_kernel_cache_file .cupy
    RunOrDie gsutil -m cp $cupy_kernel_cache_file gs://tmp-asia-pfn-public-ci/cupy-ci/
    popd
}

function Main {
    # Setup environment
    ActivateCUDA $cuda
    ActivatePython $python

    # Setup build environment variables
    $Env:CUPY_NUM_BUILD_JOBS = "16"
    $Env:CUPY_NVCC_GENERATE_CODE = "current"
    dir env:

    # Build
    RunOrDie python -m pip install Cython scipy optuna
    RunOrDie python -m pip install -e ".[all,jenkins]" -vvv

    # Import test
    RunOrDie python -c "import cupy; cupy.show_config()"

    # Unit test
    $pytest_opts = ""
    if ($test -eq "build") {
        return
    } elseif ($test -eq "test") {
        $pytest_opts = "$pytest_opts -m ""not slow"""
    } elseif ($test -eq "slow") {
        $pytest_opts = "$pytest_opts -m ""slow"""
    } else {
        throw "Unsupported test target: $target"
    }

    $use_cache = ($Env:CUPY_CI_CACHE_KERNEL -eq "1")

    if ($use_cache) {
        DownloadCache
    }
    python -m pytest -rfEX $Env:PYTEST_OPTS tests
    if (-not $?) {
        $test_retval = $LastExitCode
    }
    if ($use_cache) {
        UploadCache
    }

    if ($test_retval -ne 0) {
        throw "Test failed with status $test_retval"
    }
}

Main
