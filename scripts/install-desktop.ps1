[CmdletBinding()]
param(
    [switch]$Launch,
    [switch]$Offline,
    [switch]$DryRun,
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "NoiseReduce")
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ManifestPath = Join-Path $RepoRoot "configs\desktop_assets.json"

function Invoke-Step {
    param([string]$Description, [scriptblock]$Action)
    Write-Host "[noise-reduce] $Description"
    if (-not $DryRun) { & $Action }
}

if (-not $IsWindows) { throw "Noise Reduce desktop installer currently supports Windows only." }
if (-not (Test-Path -LiteralPath $ManifestPath)) { throw "Asset manifest not found: $ManifestPath" }

$Python = $null
foreach ($Candidate in @("py", "python")) {
    if (Get-Command $Candidate -ErrorAction SilentlyContinue) { $Python = $Candidate; break }
}
if (-not $Python) { throw "Python 3.10 or newer is required." }

if (-not $DryRun) {
    $VersionText = if ($Python -eq "py") { & py -3 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" } else { & python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" }
    $Parts = $VersionText.Trim().Split(".")
    if ([int]$Parts[0] -lt 3 -or ([int]$Parts[0] -eq 3 -and [int]$Parts[1] -lt 10)) {
        throw "Python 3.10 or newer is required; found $VersionText."
    }
}

$Venv = Join-Path $InstallDir "venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$AssetDir = Join-Path $InstallDir "assets"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$ReleaseBase = $null
if (-not $Offline) {
    # Resolve the release owner/repository from the clone when the manifest uses
    # its portable OWNER/REPOSITORY placeholder.
    try {
        $Remote = (& git -C $RepoRoot config --get remote.origin.url 2>$null).Trim()
        if ($Remote -match "github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$") {
            $ReleaseBase = "https://github.com/$($Matches[1])/$($Matches[2])"
        }
    } catch { $ReleaseBase = $null }
}

Invoke-Step "Create isolated environment at $Venv" {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    if ($Python -eq "py") { & py -3 -m venv $Venv } else { & python -m venv $Venv }
}
Invoke-Step "Install desktop package" {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install "$RepoRoot[desktop]"
}
Invoke-Step "Prepare verified model assets" {
    New-Item -ItemType Directory -Force -Path $AssetDir | Out-Null
    foreach ($Asset in $Manifest.assets) {
        $Destination = Join-Path $AssetDir $Asset.name
        if ($Offline) {
            $Source = Join-Path $RepoRoot $Asset.offline_path
            if (-not (Test-Path -LiteralPath $Source)) { throw "Offline asset missing: $Source" }
            Copy-Item -LiteralPath $Source -Destination $Destination -Force
        } else {
            $Url = [string]$Asset.url
            if ($Url -match "OWNER/REPOSITORY") {
                if (-not $ReleaseBase) {
                    throw "Release URL is not configured. Push the clone to GitHub, set remote.origin.url, or use -Offline."
                }
                $Url = $Url.Replace("https://github.com/OWNER/REPOSITORY", $ReleaseBase)
            }
            $Temporary = "$Destination.download"
            Invoke-WebRequest -Uri $Url -OutFile $Temporary
            Move-Item -LiteralPath $Temporary -Destination $Destination -Force
        }
        $Actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Asset.sha256.ToLowerInvariant()) {
            Remove-Item -LiteralPath $Destination -Force -ErrorAction SilentlyContinue
            throw "SHA-256 mismatch for $($Asset.name). Expected $($Asset.sha256); got $Actual."
        }
    }
}

$ShortcutTarget = Join-Path $Venv "Scripts\noise-reduce-desktop.exe"
Invoke-Step "Create Start Menu and Desktop shortcuts" {
    $Shell = New-Object -ComObject WScript.Shell
    $ShortcutLocations = @(
        (Join-Path ([Environment]::GetFolderPath("Desktop")) "Noise Reduce.lnk"),
        (Join-Path ([Environment]::GetFolderPath("Programs")) "Noise Reduce.lnk")
    )
    foreach ($Location in $ShortcutLocations) {
        $Shortcut = $Shell.CreateShortcut($Location)
        $Shortcut.TargetPath = $ShortcutTarget
        $Shortcut.WorkingDirectory = $InstallDir
        $Shortcut.Description = "Local speech noise reduction"
        $Shortcut.Save()
    }
}

if ($Launch) {
    Invoke-Step "Launch Noise Reduce" { Start-Process -FilePath $ShortcutTarget -WorkingDirectory $InstallDir }
}
Write-Host "[noise-reduce] Installation complete."
