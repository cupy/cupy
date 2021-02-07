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
    gsutil -m -q cp gs://tmp-asia-pfn-public-ci/cupy-ci/$cupy_kernel_cache_file .
    if (-not $?) {
        echo "*** Kernel cache unavailable"
    } else {
        RunOrDie 7z x $cupy_kernel_cache_file
        rm $cupy_kernel_cache_file
    }
    popd
}

function UploadCache {
    # Expires in 3 days, maximum 1 GB
    RunOrDie python .pfnci\trim_cupy_kernel_cache.py --expiry 259200 --max-size 1000000000 --rm

    pushd $Env:USERPROFILE
    # -mx=0 ... no compression
    # -mtc=on ... preserve timestamp
    RunOrDie 7z a -tzip -mx=0 -mtc=on $cupy_kernel_cache_file .cupy
    RunOrDie gsutil -m -q cp $cupy_kernel_cache_file gs://tmp-asia-pfn-public-ci/cupy-ci/
    popd
}

function Main {
    # Setup environment
    ActivateCUDA $cuda
    ActivatePython $python

    # Setup build environment variables
    $Env:CUPY_NUM_BUILD_JOBS = "16"
    $Env:CUPY_NVCC_GENERATE_CODE = "current"
    echo "Environment:"
    RunOrDie cmd.exe /C set

    # Build
    RunOrDie python -V
    RunOrDie python -m pip install Cython scipy optuna
    RunOrDie python -m pip freeze
    RunOrDie python -m pip install -e ".[all,jenkins]" -vvv > cupy_build.log

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
    python -m pytest -rfEX $Env:PYTEST_OPTS tests > cupy_test.log
    if (-not $?) {
        $test_retval = $LastExitCode
    }
    if ($use_cache) {
        UploadCache
    }

    # Upload test results
    echo "Uploading test results..."
    $artifact_id = $Env:CI_JOB_ID
    RunOrDie gsutil -m -q cp cupy_build.log cupy_test.log "gs://chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/"
    echo "Build Log: https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/cupy_build.log"
    echo "Test Log: https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/cupy_test.log"

    if ($test_retval -ne 0) {
        throw "Test failed with status $test_retval"
    }
}

Main
