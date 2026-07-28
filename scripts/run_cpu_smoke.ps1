param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [string]$OutputDir = "reports/generated",
    [int]$Iterations = 5000,
    [double]$SoakSeconds = 1,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv/Scripts/python.exe"
$BenchmarkOut = Join-Path $Root (Join-Path $OutputDir "cpu_benchmark.json")
$SoakOut = Join-Path $Root (Join-Path $OutputDir "cpu_soak.json")
$commands = @(
    @($Python, "scripts/benchmark_cpu.py", "--checkpoint", $Checkpoint, "--iterations", "$Iterations", "--output", $BenchmarkOut),
    @($Python, "scripts/soak_cpu.py", "--checkpoint", $Checkpoint, "--seconds", "$SoakSeconds", "--max-iterations", "$Iterations", "--output", $SoakOut)
)
if ($DryRun) {
    $commands | ForEach-Object { $_ -join " " }
    exit 0
}
foreach ($command in $commands) {
    & $command[0] $command[1..($command.Length - 1)]
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Output "CPU smoke receipts: $BenchmarkOut ; $SoakOut"
