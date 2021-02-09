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
    echo "Downloading kernel cache..."
    gsutil -m -q cp gs://tmp-asia-pfn-public-ci/cupy-ci/$cupy_kernel_cache_file .
    if (-not $?) {
        echo "*** Kernel cache unavailable"
    } else {
        echo "Extracting kernel cache..."
        RunOrDie 7z x $cupy_kernel_cache_file
        rm $cupy_kernel_cache_file
    }
    popd
}

function UploadCache {
    # Expires in 3 days, maximum 1 GB
    echo "Trimming kernel cache..."
    RunOrDie python .pfnci\trim_cupy_kernel_cache.py --expiry 259200 --max-size 1000000000 --rm

    pushd $Env:USERPROFILE
    # -mx=0 ... no compression
    # -mtc=on ... preserve timestamp
    echo "Compressing kernel cache..."
    RunOrDie 7z a -tzip -mx=0 -mtc=on $cupy_kernel_cache_file .cupy
    echo "Uploading kernel cache..."
    RunOrDie gsutil -m -q cp $cupy_kernel_cache_file gs://tmp-asia-pfn-public-ci/cupy-ci/
    popd
}

function Main {
    # Setup environment
    echo "Using CUDA $cuda and Python $python"
    ActivateCUDA $cuda
    ActivatePython $python

    # Setup build environment variables
    $Env:CUPY_NUM_BUILD_JOBS = "16"
    $Env:CUPY_NVCC_GENERATE_CODE = "current"
    echo "Environment:"
    RunOrDie cmd.exe /C set

    # Build
    echo "Setting up test environment"
    RunOrDie python -V
    RunOrDie python -m pip install Cython scipy optuna
    RunOrDie python -m pip freeze

    echo "Building..."
    RunOrDie python -m pip install -e ".[jenkins]" -vvv > cupy_build_log.txt

    # Import test
    echo "CuPy Configuration:"
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
    echo "Running test..."
    python -m pytest -rfEX $Env:PYTEST_OPTS tests > cupy_test_log.txt
    if (-not $?) {
        $test_retval = $LastExitCode
    }
    if ($use_cache) {
        UploadCache
    }

    # Upload test results
    echo "Uploading test results..."
    $artifact_id = $Env:CI_JOB_ID
    RunOrDie gsutil -m -q cp cupy_build_log.txt cupy_test_log.txt "gs://chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/"
    echo "Build Log: https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/cupy_build_log.txt"
    echo "Test Log: https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/cupy_test_log.txt"

    echo "Last 10 lines from the test output:"
    Get-Content cupy_test_log.txt -Tail 10

    if ($test_retval -ne 0) {
        throw "Test failed with status $test_retval"
    }
}

Main
