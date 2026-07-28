[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$KeepTemp
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$TempRoot = Join-Path ([IO.Path]::GetTempPath()) ("clearhop-smoke-" + [guid]::NewGuid().ToString("N"))
$WheelDir = Join-Path $TempRoot "wheelhouse"
$TargetDir = Join-Path $TempRoot "site"

try {
    New-Item -ItemType Directory -Force -Path $WheelDir, $TargetDir | Out-Null
    Write-Host "[clearhop] Build wheel"
    & $Python -m pip wheel $RepoRoot --no-deps --wheel-dir $WheelDir
    if ($LASTEXITCODE -ne 0) { throw "pip wheel failed with exit code $LASTEXITCODE." }
    $Wheel = Get-ChildItem -LiteralPath $WheelDir -Filter "*.whl" | Select-Object -First 1
    if (-not $Wheel) { throw "No wheel produced." }

    Write-Host "[clearhop] Install wheel into isolated target"
    & $Python -m pip install --no-deps --target $TargetDir $Wheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit code $LASTEXITCODE." }

    $PreviousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = $TargetDir
    try {
        & $Python -c "from src.cpu_runtime import enhance_waveform_cpu; print('runtime import: pass')"
        if ($LASTEXITCODE -ne 0) { throw "Packaged runtime import failed." }
    } finally {
        $env:PYTHONPATH = $PreviousPythonPath
    }

    Write-Host "[clearhop] Installer dry-run"
    & pwsh -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts/install-desktop.ps1") -DryRun -Offline
    if ($LASTEXITCODE -ne 0) { throw "Installer dry-run failed with exit code $LASTEXITCODE." }
    Write-Host "[clearhop] Desktop package smoke: pass"
} finally {
    if (-not $KeepTemp -and (Test-Path -LiteralPath $TempRoot)) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
    } elseif (Test-Path -LiteralPath $TempRoot) {
        Write-Host "Temporary files: $TempRoot"
    }
}
