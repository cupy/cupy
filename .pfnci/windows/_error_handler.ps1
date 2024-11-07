# https://stackoverflow.com/questions/9948517/how-to-stop-a-powershell-script-on-the-first-error

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSDefaultParameterValues['*:ErrorAction']='Stop'

function RunOrDie {
    $cmd, $params = $args
    $params = @($params)
    $global:LastExitCode = 0
    & $cmd @params
    if (-not $?) {
        throw "Command failed (exit code = $LastExitCode): $args"
    }
}

function RunWithTimeout {
    param(
        [Parameter(Mandatory=$true)]
        [int]$timeout,
        [Parameter(Mandatory=$true)]
        [string]$output,
        [Parameter(Mandatory=$true)]
        [string]$command,
        [Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)]
        [string[]]$params
    )
    $process = Start-Process -PassThru -NoNewWindow -RedirectStandardOutput $output -FilePath $command -ArgumentList $params
    try {
        $process | Wait-Process -Timeout $timeout
    } catch [TimeoutException] {
        Write-Warning "Command timed out: $command $params"
        $process | Stop-Process -Force
        if (!$process.HasExited) {
            Write-Warning "Failed to force terminate the process: $command $params"
            return 999  # ExitCode unavailable, return a dummy value
        }
    }

    # Return code will be -1 when force terminated.
    return $process.ExitCode
}
