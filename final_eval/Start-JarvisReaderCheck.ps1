$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$outputPath = Join-Path $projectRoot 'outputs\final_eval\target_calibration\gptoss_fallback_v1'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project .venv Python is missing.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'jarvis.env'))) { throw 'Fill final_eval/jarvis.env first.' }
if (Test-Path -LiteralPath (Join-Path $outputPath 'access_check.lock')) { throw 'Fallback check lock exists; check previous process before restarting.' }
$processPath = Join-Path $outputPath 'process.json'
if (Test-Path -LiteralPath $processPath) {
    $previous = Get-Content -LiteralPath $processPath -Raw | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) { throw 'Previous fallback PID is still active.' }
}
Push-Location $projectRoot
try {
    & $pythonPath -m final_eval.jarvis_reader_check --dry-run
    if ($LASTEXITCODE -ne 0) { throw 'Fallback preflight failed; nothing launched.' }
} finally { Pop-Location }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$stdoutPath = Join-Path $outputPath 'reader.stdout.log'
$stderrPath = Join-Path $outputPath 'reader.stderr.log'
$child = Start-Process -FilePath $pythonPath -ArgumentList '-u -m final_eval.jarvis_reader_check' `
    -WorkingDirectory $projectRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
@{ pid = $child.Id; started_utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath $processPath -Encoding UTF8
Write-Host "Started six-case GPT-OSS fallback check. PID: $($child.Id)"
Write-Host "Progress: $stdoutPath"
Write-Host "Errors:   $stderrPath"
Write-Host 'Maximum eight cumulative HTTP attempts. Keep laptop awake/online; no full pilot starts automatically.'
