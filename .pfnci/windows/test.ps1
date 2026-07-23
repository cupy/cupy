Param(
    [String]$cuda,
    [String]$python,
    [String]$numpy,
    [String]$scipy,
    [String]$test
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

. "$PSScriptRoot\_flexci.ps1"


function DownloadCache([String]$gcs_dir, [String]$cupy_kernel_cache_file) {
    pushd $Env:USERPROFILE
    echo "Downloading kernel cache..."
    gsutil -m -q cp "$gcs_dir/$cupy_kernel_cache_file" .
    if (-not $?) {
        echo "*** Kernel cache unavailable ($gcs_dir/$cupy_kernel_cache_file)"
    } else {
        echo "Extracting kernel cache..."
        RunOrDie 7z x -aoa $cupy_kernel_cache_file
        rm $cupy_kernel_cache_file
    }
    popd
}

function UploadCache([String]$gcs_dir, [String]$cupy_kernel_cache_file) {
    # Limit *.cubin cache by total size.
    # Note: --expiry cannot be used as access time is not updated on Windows.
    # TODO: Also clean up ~/.cupy/callback_cache/*.ltoir
    echo "Trimming kernel cache..."
    RunOrDie python .pfnci\trim_cupy_kernel_cache.py --max-size 10737418240 --rm

    pushd $Env:USERPROFILE
    # -mx=0 ... no compression
    # -mtc=on ... preserve timestamp
    echo "Compressing kernel cache..."
    RunOrDie 7z a -tzip -mx=0 -mtc=on $cupy_kernel_cache_file .cupy
    echo "Uploading kernel cache..."
    RunOrDie gsutil -m -q cp $cupy_kernel_cache_file $gcs_dir/
    popd
}

function PublishTestResults {
    # Upload test results
    echo "Uploading test results..."
    $artifact_id = $Env:CI_JOB_ID
    RunOrDie gsutil -m -q cp cupy_build_log.txt cupy_test_log.txt "gs://chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/"
    echo "Build Log:"
    echo "https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/cupy_build_log.txt"
    echo "Test Log:"
    echo "https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/cupy_test_log.txt"
}

function Main {
    PrioritizeFlexCIDaemon
    EnableLongPaths

    # Enable symbolic links and re-checkout
    git config core.symlinks true
    git reset --hard

    # Setup environment
    echo "Using CUDA $cuda and Python $python"
    ActivateCUDA $cuda
    ActivatePython $python

    # Setup build environment variables
    $Env:CUPY_NUM_BUILD_JOBS = "16"
    $Env:CUPY_NVCC_GENERATE_CODE = "current"
    echo "Environment:"
    # Redact the wheel-fetch token (provisioned in the FlexCI job env) from the log.
    cmd.exe /C set | Where-Object { $_ -notmatch '^(CUPY_CI_GITHUB_TOKEN|GH_TOKEN)=' }

    # Build
    echo "Setting up test environment"
    RunOrDie python -V
    RunOrDie python -m pip install -U pip setuptools wheel
    RunOrDie python -m pip install -U google-cloud-storage  # For GCP kernel cache backend
    RunOrDie python -m pip freeze

    echo "Building..."
    $build_retval = 0
    RunOrDie python -m pip install "numpy==$numpy.*" "scipy==$scipy.*" "Cython==3.2.*,!=3.2.6"

    # Fetch the CuPy wheel built by GHA (.github/workflows/ci.yml -> build-wheel.yml)
    # for the commit under test, then pip-install it. Artifacts are
    # producer-pinned: only artifacts uploaded by a successful pull_request or
    # push run of ci.yml at the exact commit under test are accepted. Failures
    # at this step mean either the wheel-build matrix does not cover this
    # (CUDA, Python) tuple (by design, until it is expanded), or a fresh /test
    # needs to be issued.
    if (-not $Env:CUPY_CI_GITHUB_TOKEN) {
        throw "CUPY_CI_GITHUB_TOKEN env var is required to fetch wheel artifacts from cupy/cupy CI"
    }
    $Env:GH_TOKEN = $Env:CUPY_CI_GITHUB_TOKEN
    $py_ver = (& python -c "import sys, sysconfig; print(f'{sys.version_info.major}.{sys.version_info.minor}' + ('t' if sysconfig.get_config_var('Py_GIL_DISABLED') else ''))").Trim()
    $cuda_major = $cuda.Split(".")[0]

    # Derive the artifact suffix ci-guard.sh emits:
    #   PR (pull_request labeled):  pr<N>-<head sha>
    #   push (post-merge):          <sha>
    # FLEXCI_BRANCH is refs/pull/<N>/head on PR tests, else a branch ref.
    if (IsPullRequestTest) {
        $pr_number = (GetPullRequestNumber)
        $suffix_prefix = "pr${pr_number}-"
        $event = "pull_request"
    } else {
        $suffix_prefix = ""
        $event = "push"
    }

    # The FlexCI checkout may be the tested commit directly, or a merge of that
    # commit into its base -- in the merge case the tested commit is the second
    # parent.
    $candidate_shas = @((& git rev-parse HEAD).Trim())
    $second_parent = (& git rev-parse --verify --quiet "HEAD^2")
    if ($LASTEXITCODE -eq 0 -and $second_parent) {
        $candidate_shas += $second_parent.Trim()
    }

    $artifact_id = $null
    $run_id = $null
    foreach ($sha in $candidate_shas) {
        $expected_name = "cupy-cuda${cuda_major}x-py${py_ver}-win-64-${suffix_prefix}${sha}"

        # The artifact name is unique per (PR, commit, platform, Python), so
        # query it directly -- each match carries its producing run id, which is
        # our Run ID (no run enumeration). gh failure is fatal.
        $rows = (& gh api --paginate "repos/cupy/cupy/actions/artifacts?name=${expected_name}&per_page=100" --jq '.artifacts[] | select(.expired == false) | "\(.workflow_run.head_sha) \(.id) \(.workflow_run.id)"')
        if ($LASTEXITCODE -ne 0) { throw "Failed to query wheel artifacts for ${sha}" }
        foreach ($row in $rows) {
            $parts = "$row".Trim() -split '\s+'
            if ($parts[0] -ne $sha) { continue }
            $cand_artifact = $parts[1]
            $cand_run = $parts[2]
            # Producer pin: confirm the producing run is a successful ci.yml run
            # for the expected event (the name is a free string any run could
            # pick); this also resolves the Run ID with no guessing.
            $meta = (& gh api "repos/cupy/cupy/actions/runs/${cand_run}" --jq '"\(.path) \(.event) \(.conclusion)"')
            if ($LASTEXITCODE -ne 0) { throw "Failed to query producing run ${cand_run}" }
            $m = "$meta".Trim() -split '\s+'
            if ($m[0] -eq ".github/workflows/ci.yml" -and $m[1] -eq $event -and $m[2] -eq "success") {
                $artifact_id = $cand_artifact
                $run_id = $cand_run
                break
            }
        }
        if ($artifact_id) { break }
    }
    if (-not $artifact_id) {
        throw "No wheel artifact from a successful ci.yml run for candidate SHAs: $($candidate_shas -join ', ') (name: cupy-cuda${cuda_major}x-py${py_ver}-win-64-${suffix_prefix}<sha>). Re-issue /test on the PR."
    }
    Write-Output "Resolved wheel artifact $artifact_id from ci.yml run $run_id"

    $wheel_dir = New-Item -ItemType Directory -Path ([System.IO.Path]::GetTempFileName() + ".d") -Force
    $zip_path = Join-Path $wheel_dir.FullName "artifact.zip"
    # Route the binary download through cmd.exe -- PowerShell's `>` uses
    # UTF-16 encoding by default, which would corrupt the zip stream.
    # Download is by immutable artifact ID.
    $cmd = "gh api `"repos/cupy/cupy/actions/artifacts/${artifact_id}/zip`" > `"${zip_path}`""
    & cmd.exe /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "gh api download of artifact ${artifact_id} failed (exit $LASTEXITCODE)" }
    Expand-Archive -Path $zip_path -DestinationPath $wheel_dir.FullName -Force
    Remove-Item $zip_path
    # Drop the token from the environment before any PR-controlled code (pip
    # install / pytest) runs, so the test process cannot read it back.
    Remove-Item Env:GH_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:CUPY_CI_GITHUB_TOKEN -ErrorAction SilentlyContinue

    $wheel = (Get-ChildItem -Path $wheel_dir.FullName -Filter '*.whl' | Select-Object -First 1).FullName
    if (-not $wheel) { throw "No wheel found under $($wheel_dir.FullName)" }

    python -m pip install -v "${wheel}[all,test]" > cupy_build_log.txt
    if (-not $?) {
        $build_retval = $LastExitCode
    }

    echo "------------------------------------------------------------------------------------------"
    echo "Last 10 lines from the build output:"
    Get-Content cupy_build_log.txt -Tail 10
    echo "------------------------------------------------------------------------------------------"

    if ($build_retval -ne 0) {
        echo "n/a" > cupy_test_log.txt
        PublishTestResults
        throw "Build failed with status $build_retval"
    }

    $Env:CUPY_TEST_GPU_LIMIT = $Env:GPU
    $Env:CUPY_DUMP_CUDA_SOURCE_ON_ERROR = "1"
    $Env:CUPY_NVRTC_USE_PCH = "1"

    # Unit test
    if ($test -eq "build") {
        return
    } elseif ($test -eq "test") {
        $pytest_opts = "-m", '"not slow"'
    } elseif ($test -eq "slow") {
        $pytest_opts = "-m", "slow"
    } else {
        throw "Unsupported test target: $target"
    }

    $base_branch = (Get-Content .pfnci\BRANCH)
    $is_pull_request = IsPullRequestTest
    $cache_archive = "windows-cuda${cuda}-${base_branch}.zip"
    $cache_gcs_dir = "gs://tmp-asia-pfn-public-ci/cupy-ci/cache"
    $cache_pr_gcs_dir = "${cache_gcs_dir}-pr-" + (GetPullRequestNumber)

    #DownloadCache "${cache_gcs_dir}" "${cache_archive}"
    #if ($is_pull_request) {
    #    DownloadCache "${cache_pr_gcs_dir}" "${cache_archive}"
    #}

    $Env:CUPY_CI_ENABLE_GCP_KERNEL_CACHE = "1"

    if (-Not $is_pull_request) {
        $Env:CUPY_TEST_FULL_COMBINATION = "1"
    }
    # Skip full test for these CUDA versions as compilation seems so slow
    if (($cuda -eq "12.0") -or ($cuda -eq "12.1") -or ($cuda -eq "12.2")) {
        $Env:CUPY_TEST_FULL_COMBINATION = "0"
    }

    pushd tests
    echo "CuPy Configuration:"
    RunOrDie python -c "import cupy; print(cupy); cupy.show_config()"
    echo "Running test..."
    # TODO(leofang): allow larger/adjustable timeout?
    $test_retval = RunWithTimeout -timeout 18000 -output ../cupy_test_log.txt -- python -m pytest -rfEX @pytest_opts .
    popd

    #if ($is_pull_request) {
    #    UploadCache "${cache_pr_gcs_dir}" "${cache_archive}"
    #} else {
    #    UploadCache "${cache_gcs_dir}" "${cache_archive}"
    #}

    echo "------------------------------------------------------------------------------------------"
    echo "Last 10 lines from the test output:"
    Get-Content cupy_test_log.txt -Tail 10
    echo "------------------------------------------------------------------------------------------"

    PublishTestResults
    if ($test_retval -ne 0) {
        throw "Test failed with status $test_retval"
    }
}

Main
