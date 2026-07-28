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

    $OutputRoot = if ([IO.Path]::IsPathRooted($OutputDir)) {
        [IO.Path]::GetFullPath($OutputDir)
    } else {
        [IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputDir))
    }
    $ReleaseRoot = Join-Path $OutputRoot "ClearHop"
    if (-not (Test-Path -LiteralPath $ReleaseRoot -PathType Container)) {
        throw "PyInstaller output directory not found: $ReleaseRoot"
    }
    $LicenseSource = Join-Path $RepoRoot "LICENSE"
    $LicenseTarget = Join-Path $ReleaseRoot "LICENSE"
    Copy-Item -LiteralPath $LicenseSource -Destination $LicenseTarget -Force
    $SourceLicenseHash = (Get-FileHash -LiteralPath $LicenseSource -Algorithm SHA256).Hash
    $TargetLicenseHash = (Get-FileHash -LiteralPath $LicenseTarget -Algorithm SHA256).Hash
    if ($SourceLicenseHash -ne $TargetLicenseHash) {
        throw "Packaged license hash mismatch."
    }
} finally {
    Pop-Location
}
