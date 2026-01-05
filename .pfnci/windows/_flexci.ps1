function ActivatePython($version) {
    if ($version -eq "3.10") {
        $pydir = "Python310"
    } elseif ($version -eq "3.11") {
        $pydir = "Python311"
    } elseif ($version -eq "3.12") {
        $pydir = "Python312"
    } elseif ($version -eq "3.13") {
        $pydir = "Python313"
    } elseif ($version -eq "3.14") {
        $pydir = "Python314"
    } else {
        throw "Unsupported Python version: $version"
    }
    $Env:PATH = "C:\Development\Python\$pydir;C:\Development\Python\$pydir\Scripts;" + $Env:PATH
}

function ActivateCUDA($version) {
    if ($version -eq "12.0") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_0
    } elseif ($version -eq "12.1") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_1
    } elseif ($version -eq "12.2") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_2
    } elseif ($version -eq "12.3") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_3
    } elseif ($version -eq "12.4") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_4
    } elseif ($version -eq "12.5") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_5
    } elseif ($version -eq "12.6") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_6
    } elseif ($version -eq "12.8") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_8
    } elseif ($version -eq "12.9") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_9
    } elseif ($version -eq "12.x") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_9
    } elseif ($version -eq "13.0") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V13_0
    } elseif ($version -eq "13.x") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V13_0
    } else {
        throw "Unsupported CUDA version: $version"
    }
    $Env:PATH = "$Env:CUDA_PATH\bin;$Env:CUDA_PATH\bin\x64;" + $Env:PATH
}

function IsPullRequestTest() {
    return ${Env:FLEXCI_BRANCH} -ne $null -and ${Env:FLEXCI_BRANCH}.StartsWith("refs/pull/")
}

function GetPullRequestNumber() {
    if (${Env:FLEXCI_BRANCH} -match "refs/pull/(\d+)/") {
        return $matches[1]
    }
    return 0
}

function PrioritizeFlexCIDaemon() {
    echo "Prioritizing FlexCI daemon process..."
    wmic.exe process where 'name="imosci.exe"' CALL setpriority realtime
    if (-not $?) {
        throw "Failed to change priority of daemon (exit code = $LastExitCode)"
    }
}

function EnableLongPaths() {
    Set-ItemProperty "Registry::HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -value 1
}
