[CmdletBinding()]
param(
    [string]$OutputDir = "dist",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
if (-not $IsWindows) { throw "Desktop release builds currently support Windows only." }
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
Push-Location $RepoRoot
try {
    $Args = @(
        "--noconfirm", "--onedir", "--windowed",
        "--name", "ClearHop",
        "--add-data", "artifacts/cpu_bundle/checkpoint.pth;assets",
        "--distpath", $OutputDir,
        "desktop/app.py"
    )
    if ($Clean) { $Args = @("--clean") + $Args }
    & $Python -m PyInstaller @Args
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}
