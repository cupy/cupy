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

function PublishTestResults {
    # Upload test results
    echo "Uploading test results..."
    $artifact_id = $Env:CI_JOB_ID
    RunOrDie gsutil -m -q cp cupy_build_log.txt cupy_test_log.txt "gs://chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/"
    echo "Build Log: https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/cupy_build_log.txt"
    echo "Test Log: https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/cupy_test_log.txt"
}

function Main {
    PrioritizeFlexCIDaemon

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
    $build_retval = 0
    python -m pip install -e ".[all,jenkins]" -vvv > cupy_build_log.txt
    if (-not $?) {
        $build_retval = $LastExitCode
    }
    echo "Last 10 lines from the build output:"
    Get-Content cupy_build_log.txt -Tail 10

    if ($build_retval -ne 0) {
        echo "n/a" > cupy_test_log.txt
        PublishTestResults
        throw "Build failed with status $build_retval"
    }

    # Import test
    echo "CuPy Configuration:"
    RunOrDie python -c "import cupy; cupy.show_config()"

    # Unit test
    if ($test -eq "build") {
        return
    } elseif ($test -eq "test") {
        $pytest_opts = "-m", "not slow"
    } elseif ($test -eq "slow") {
        $pytest_opts = "-m", "slow"
    } else {
        throw "Unsupported test target: $target"
    }

    $use_cache = ($Env:CUPY_CI_CACHE_KERNEL -eq "1")
    $upload_cache = -Not (IsPullRequestTest)

    if ($use_cache) {
        DownloadCache
    }
    echo "Running test..."
    $test_retval = 0
    python -c "import cupy; cupy.show_config()" > cupy_test_log.txt
    python -m pytest -rfEX @pytest_opts tests >> cupy_test_log.txt
    if (-not $?) {
        $test_retval = $LastExitCode
    }
    if ($use_cache -And $upload_cache) {
        UploadCache
    }

    echo "Last 10 lines from the test output:"
    Get-Content cupy_test_log.txt -Tail 10

    PublishTestResults
    if ($test_retval -ne 0) {
        throw "Test failed with status $test_retval"
    }
}

Main
