[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [string]$WorkingDirectory = $env:TEMP
)
$ErrorActionPreference = "Stop"
$exe = (Resolve-Path -LiteralPath $Executable).Path
$work = (Resolve-Path -LiteralPath $WorkingDirectory).Path
if ((Split-Path -Parent $exe) -eq (Get-Location).Path) { throw "Packaged smoke must run outside the repository." }
$proc = Start-Process -FilePath $exe -WorkingDirectory $work -PassThru
Start-Sleep -Seconds 3
if ($proc.HasExited -and $proc.ExitCode -ne 0) { throw "Packaged executable exited with $($proc.ExitCode)." }
if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Write-Output "packaged launch: pass"
