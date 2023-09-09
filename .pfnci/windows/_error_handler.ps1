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
    $process = Start-Process -NoNewWindow -PassThru -RedirectStandardOutput $output -FilePath $command -ArgumentList $params

    for ($i = 0; $i -lt $timeout; $i++) {
        if ($process.HasExited) {
            break
        }
        Start-Sleep -Seconds 1
    }
    if (!$process.HasExited) {
        Write-Warning "Command timed out: $command $params"
        $process | Stop-Process -Force

        # Wait until the process exit to retrieve the exit code.
        for ($i = 0; $i -lt 100; $i++) {
            if ($process.HasExited) {
                break
            }
            Start-Sleep -Seconds 0.3
        }
        if (!$process.HasExited) {
            Write-Warning "Failed to stop the process: $command $params"
            return 999  # ExitCode unavailable, return a dummy value
        }
    }

    return $process.ExitCode
}
