[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [string]$WorkingDirectory = $env:TEMP,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedCheckpointSha256,
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
$exe = (Resolve-Path -LiteralPath $Executable).Path
$work = (Resolve-Path -LiteralPath $WorkingDirectory).Path
$smoke = Join-Path $work "clearhop-packaged-smoke-$PID"
New-Item -ItemType Directory -Path $smoke -ErrorAction Stop | Out-Null
$inputWav = Join-Path $smoke "input.wav"
$outputWav = Join-Path $smoke "output.wav"
$receiptPath = Join-Path $smoke "receipt.json"

# Generate a publishable one-second PCM fixture without repository dependencies.
$sampleRate = 16000
$samples = New-Object Int16[] $sampleRate
for ($i = 0; $i -lt $samples.Length; $i++) {
    $samples[$i] = [int16](3000.0 * [Math]::Sin(2.0 * [Math]::PI * 440.0 * $i / $sampleRate))
}
$stream = [IO.File]::Open($inputWav, [IO.FileMode]::CreateNew)
$writer = [IO.BinaryWriter]::new($stream)
try {
    $writer.Write([Text.Encoding]::ASCII.GetBytes("RIFF")); $writer.Write([int](36 + 2 * $samples.Length))
    $writer.Write([Text.Encoding]::ASCII.GetBytes("WAVEfmt ")); $writer.Write([int]16)
    $writer.Write([int16]1); $writer.Write([int16]1); $writer.Write([int]$sampleRate)
    $writer.Write([int](2 * $sampleRate)); $writer.Write([int16]2); $writer.Write([int16]16)
    $writer.Write([Text.Encoding]::ASCII.GetBytes("data")); $writer.Write([int](2 * $samples.Length))
    foreach ($sample in $samples) { $writer.Write($sample) }
} finally {
    $writer.Dispose(); $stream.Dispose()
}

$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $exe
$start.WorkingDirectory = $work
$start.UseShellExecute = $false
$start.Environment.Remove("NOISE_REDUCE_CHECKPOINT")
foreach ($argument in @("--smoke-input", $inputWav, "--smoke-output", $outputWav, "--smoke-receipt", $receiptPath)) {
    $start.ArgumentList.Add($argument)
}
$process = [Diagnostics.Process]::Start($start)
if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
    Stop-Process -Id $process.Id -Force
    throw "Packaged denoise timed out after $TimeoutSeconds seconds."
}
if ($process.ExitCode -ne 0) { throw "Packaged denoise exited with $($process.ExitCode)." }
if (-not (Test-Path -LiteralPath $outputWav) -or -not (Test-Path -LiteralPath $receiptPath)) {
    throw "Packaged denoise did not create output WAV and receipt."
}
$receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
$actualOutputHash = (Get-FileHash -LiteralPath $outputWav -Algorithm SHA256).Hash.ToLowerInvariant()
if ($receipt.status -ne "pass" -or $receipt.output_sha256 -ne $actualOutputHash) {
    throw "Packaged receipt is not pass/hash-bound."
}
if (
    [int]$receipt.input_samples -le 0 -or
    [int]$receipt.output_samples -ne [int]$receipt.input_samples -or
    $receipt.checkpoint_sha256 -ne $ExpectedCheckpointSha256.ToLowerInvariant()
) {
    throw "Packaged receipt lacks model-bound denoise evidence."
}
Write-Output "packaged denoise: pass; receipt=$receiptPath"
