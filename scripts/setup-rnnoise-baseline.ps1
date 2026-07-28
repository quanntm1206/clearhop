param(
    [string]$OutputDir = ".artifacts/external_baselines/rnnoise",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath = Join-Path $root "configs/rnnoise_toolchain.json"
$dockerfilePath = Join-Path $root "containers/rnnoise/Dockerfile"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

if ($manifest.schema_version -ne 1 -or $manifest.base_image -notmatch '^ubuntu@sha256:[0-9a-f]{64}$') {
    throw "blocked: invalid pinned RNNoise toolchain manifest"
}
if ($manifest.source_commit -notmatch '^[0-9a-f]{40}$' -or $manifest.source_archive_sha256 -notmatch '^[0-9a-f]{64}$') {
    throw "blocked: invalid pinned RNNoise source provenance"
}
if ($manifest.model_archive_sha256 -notmatch '^[0-9a-f]{64}$' -or $manifest.model_archive_url -notmatch '^https://') {
    throw "blocked: invalid pinned RNNoise model provenance"
}
if ($manifest.apt_snapshot -notmatch '^https://snapshot\.ubuntu\.com/ubuntu/[0-9]{8}T[0-9]{6}Z/$') {
    throw "blocked: invalid pinned Ubuntu apt snapshot"
}

$commit = [string]$manifest.source_commit
$output = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDir))
$tag = "clearhop-rnnoise-builder:$($commit.Substring(0, 12))"
$arguments = @(
    "build", "--pull=false", "--platform", [string]$manifest.platform,
    "--file", $dockerfilePath, "--tag", $tag,
    "--build-arg", "BASE_IMAGE=$($manifest.base_image)",
    "--build-arg", "RNNOISE_COMMIT=$commit",
    "--build-arg", "RNNOISE_ARCHIVE_URL=$($manifest.source_archive_url)",
    "--build-arg", "RNNOISE_ARCHIVE_SHA256=$($manifest.source_archive_sha256)",
    "--build-arg", "RNNOISE_MODEL_URL=$($manifest.model_archive_url)",
    "--build-arg", "RNNOISE_MODEL_SHA256=$($manifest.model_archive_sha256)",
    "--build-arg", "APT_SNAPSHOT_URL=$($manifest.apt_snapshot)",
    "--build-arg", "AUTOCONF_VERSION=$($manifest.packages.autoconf)",
    "--build-arg", "AUTOMAKE_VERSION=$($manifest.packages.automake)",
    "--build-arg", "BUILD_ESSENTIAL_VERSION=$($manifest.packages.'build-essential')",
    "--build-arg", "CA_CERTIFICATES_VERSION=$($manifest.packages.'ca-certificates')",
    "--build-arg", "CURL_VERSION=$($manifest.packages.curl)",
    "--build-arg", "LIBTOOL_VERSION=$($manifest.packages.libtool)",
    "--build-arg", "PKG_CONFIG_VERSION=$($manifest.packages.'pkg-config')",
    $root
)

if ($DryRun) {
    [pscustomobject]@{
        status = "dry-run"
        pinned_commit = $commit
        base_image = $manifest.base_image
        source_archive_sha256 = $manifest.source_archive_sha256
        model_archive_sha256 = $manifest.model_archive_sha256
        toolchain_manifest = $manifestPath
        output = $output
        command = "docker " + ($arguments -join " ")
    } | ConvertTo-Json -Depth 3
    exit 0
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "blocked: docker executable unavailable; rerun with Docker or in Linux CI"
}

& docker @arguments
if ($LASTEXITCODE -ne 0) {
    throw "blocked: pinned RNNoise container build failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Path $output -Force | Out-Null
& docker run --rm --volume "${output}:/export" $tag sh -c "cp -a /out/. /export/"
if ($LASTEXITCODE -ne 0) {
    throw "blocked: unable to copy RNNoise build outputs"
}

foreach ($name in "rnnoise_demo", "SHA256SUMS", "pinned_commit.txt", "COPYING", "README", "toolchain_packages.txt") {
    if (-not (Test-Path -LiteralPath (Join-Path $output $name))) {
        throw "blocked: RNNoise build omitted $name"
    }
}
if ((Get-Content -LiteralPath (Join-Path $output "pinned_commit.txt") -Raw).Trim() -ne $commit) {
    throw "blocked: RNNoise output commit mismatch"
}
$installed = @{}
foreach ($line in Get-Content -LiteralPath (Join-Path $output "toolchain_packages.txt")) {
    $parts = $line -split '=', 2
    if ($parts.Count -eq 2) {
        $installed[$parts[0]] = $parts[1]
    }
}
foreach ($package in $manifest.packages.PSObject.Properties) {
    if ($installed[$package.Name] -ne [string]$package.Value) {
        throw "blocked: toolchain package mismatch for $($package.Name)"
    }
}
Get-Content -LiteralPath (Join-Path $output "SHA256SUMS")
