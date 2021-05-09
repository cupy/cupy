function ActivatePython($version) {
    if ($version -eq "3.5") {
        $pydir = "Python35"
    } elseif ($version -eq "3.6") {
        $pydir = "Python36"
    } elseif ($version -eq "3.7") {
        $pydir = "Python37"
    } elseif ($version -eq "3.8") {
        $pydir = "Python38"
    } elseif ($version -eq "3.9") {
        $pydir = "Python39"
    } else {
        throw "Unsupported Python version: $version"
    }
    $Env:PATH = "C:\Development\Python\$pydir;C:\Development\Python\$pydir\Scripts;" + $Env:PATH
}

function ActivateCUDA($version) {
    if ($version -eq "8.0") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V8_0
    } elseif ($version -eq "9.0") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V9_0
    } elseif ($version -eq "9.1") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V9_1
    } elseif ($version -eq "9.2") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V9_2
    } elseif ($version -eq "10.0") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V10_0
    } elseif ($version -eq "10.1") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V10_1
    } elseif ($version -eq "10.2") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V10_2
    } elseif ($version -eq "11.0") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_0
    } elseif ($version -eq "11.1") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_1
    } elseif ($version -eq "11.2") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_2
    } else {
        throw "Unsupported CUDA version: $version"
    }
    $Env:PATH = "$Env:CUDA_PATH\bin;$Env:ProgramFiles\NVIDIA Corporation\NvToolsExt\bin\x64;" + $Env:PATH
}

function IsPullRequestTest() {
    return ${Env:FLEXCI_BRANCH} -ne $null -and ${Env:FLEXCI_BRANCH}.StartsWith("refs/pull/")
}

function PrioritizeFlexCIDaemon() {
    echo "Prioritizing FlexCI daemon process..."
    wmic.exe process where 'name="imosci.exe"' CALL setpriority realtime
    if (-not $?) {
        throw "Failed to change priority of daemon (exit code = $LastExitCode)"
    }
}
