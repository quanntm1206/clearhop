[CmdletBinding()]
param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    & $Python scripts/verify.py --publish-readiness --output reports/generated/publish_readiness.json
    if ($LASTEXITCODE -ne 0) { throw "Publish-readiness audit failed." }
} finally {
    Pop-Location
}
