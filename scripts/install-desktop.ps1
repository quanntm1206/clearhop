[CmdletBinding()]
param(
    [switch]$Launch,
    [switch]$Offline,
    [switch]$DryRun,
    [switch]$NoShortcuts,
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "ClearHop")
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ManifestPath = Join-Path $RepoRoot "configs\desktop_assets.json"
$ConstraintPath = Join-Path $RepoRoot "constraints\release.txt"

function Invoke-Step {
    param([string]$Description, [scriptblock]$Action)
    Write-Host "[clearhop] $Description"
    if (-not $DryRun) { & $Action }
}

if (-not $IsWindows) { throw "ClearHop desktop installer currently supports Windows only." }
if (-not (Test-Path -LiteralPath $ManifestPath)) { throw "Asset manifest not found: $ManifestPath" }
if (-not (Test-Path -LiteralPath $ConstraintPath)) { throw "Release constraints not found: $ConstraintPath" }
$ConstraintUri = ([Uri](Resolve-Path -LiteralPath $ConstraintPath).Path).AbsoluteUri

$Python = $null
$PythonPrefix = @()
if (Get-Command "py" -ErrorAction SilentlyContinue) {
    if ($DryRun) {
        $Python = "py"
        $PythonPrefix = @("-3.11")
    } else {
        & py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $Python = "py"
            $PythonPrefix = @("-3.11")
        }
    }
}
if (-not $Python -and (Get-Command "python" -ErrorAction SilentlyContinue)) {
    if ($DryRun) {
        $Python = "python"
    } else {
        & python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
        if ($LASTEXITCODE -eq 0) { $Python = "python" }
    }
}
if (-not $Python) {
    throw "Python 3.11 is required by the verified desktop dependency lock. Install Python 3.11, then rerun this command."
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
    & $Python @PythonPrefix -m venv $Venv
}
Invoke-Step "Install desktop package" {
    # Absolute inheritance also constrains PEP 517 build-isolation subprocesses.
    # The file URI keeps paths containing spaces as one pip option value.
    $env:PIP_CONSTRAINT = $ConstraintUri
    & $VenvPython -m pip install "pip==26.1.2" "setuptools==83.0.0"
    if ($LASTEXITCODE -ne 0) { throw "Pinned packaging bootstrap failed." }
    & $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu"
    if ($LASTEXITCODE -ne 0) { throw "CPU-only PyTorch installation failed." }
    & $VenvPython -c "import torch; assert torch.version.cuda is None"
    if ($LASTEXITCODE -ne 0) { throw "Installed PyTorch runtime is not CPU-only." }
    & $VenvPython -m pip install -c $ConstraintPath "$RepoRoot[desktop]"
    if ($LASTEXITCODE -ne 0) { throw "ClearHop desktop package installation failed." }
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
if (-not $NoShortcuts) {
    Invoke-Step "Create Start Menu and Desktop shortcuts" {
        $Shell = New-Object -ComObject WScript.Shell
        $ShortcutLocations = @(
            (Join-Path ([Environment]::GetFolderPath("Desktop")) "ClearHop.lnk"),
            (Join-Path ([Environment]::GetFolderPath("Programs")) "ClearHop.lnk")
        )
        foreach ($Location in $ShortcutLocations) {
            $Shortcut = $Shell.CreateShortcut($Location)
            $Shortcut.TargetPath = $ShortcutTarget
            $Shortcut.WorkingDirectory = $InstallDir
            $Shortcut.Description = "ClearHop local speech denoising"
            $Shortcut.Save()
        }
    }
}

if ($Launch) {
    Invoke-Step "Launch ClearHop" { Start-Process -FilePath $ShortcutTarget -WorkingDirectory $InstallDir }
}
Write-Host "[clearhop] Installation complete."
