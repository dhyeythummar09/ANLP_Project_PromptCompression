$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$outputPath = Join-Path $projectRoot 'outputs\final_eval\target_calibration\deepseek_v1'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project .venv Python is missing.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'jarvis.env'))) { throw 'Fill final_eval/jarvis.env first.' }
if (Test-Path -LiteralPath (Join-Path $outputPath 'access_check.lock')) {
    throw 'Calibration lock exists. Confirm the previous process stopped before removing a stale lock.'
}
$processPath = Join-Path $outputPath 'process.json'
if (Test-Path -LiteralPath $processPath) {
    $previous = Get-Content -LiteralPath $processPath -Raw | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) { throw 'Previous calibration PID is still active.' }
}
Push-Location $projectRoot
try {
    & $pythonPath -m final_eval.jarvis_calibrate --dry-run
    if ($LASTEXITCODE -ne 0) { throw 'Calibration preflight failed; nothing launched.' }
} finally { Pop-Location }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$stdoutPath = Join-Path $outputPath 'calibration.stdout.log'
$stderrPath = Join-Path $outputPath 'calibration.stderr.log'
$child = Start-Process -FilePath $pythonPath -ArgumentList '-u -m final_eval.jarvis_calibrate' `
    -WorkingDirectory $projectRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
@{ pid = $child.Id; started_utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath $processPath -Encoding UTF8
Write-Host "Started small DeepSeek settings test. PID: $($child.Id)"
Write-Host "Progress: $stdoutPath"
Write-Host "Errors:   $stderrPath"
Write-Host 'Maximum 16 HTTP attempts across both settings and resumptions. No full pilot starts automatically.'
Write-Host 'Keep the laptop awake and online.'
