$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$outputPath = Join-Path $projectRoot 'outputs\final_eval\target_calibration\qwen_cap8192_v1'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project .venv Python is missing.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'jarvis.env'))) { throw 'Fill final_eval/jarvis.env first.' }
if (Test-Path -LiteralPath (Join-Path $outputPath 'access_check.lock')) { throw 'Cap check lock exists. Confirm previous process stopped.' }
$processPath = Join-Path $outputPath 'process.json'
if (Test-Path -LiteralPath $processPath) {
    $previous = Get-Content -LiteralPath $processPath -Raw | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) { throw 'Previous cap-check PID is still active.' }
}
Push-Location $projectRoot
try {
    & $pythonPath -m final_eval.jarvis_qwen_cap --dry-run
    if ($LASTEXITCODE -ne 0) { throw 'Qwen cap-check preflight failed; nothing launched.' }
} finally { Pop-Location }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$stdoutPath = Join-Path $outputPath 'cap.stdout.log'
$stderrPath = Join-Path $outputPath 'cap.stderr.log'
$child = Start-Process -FilePath $pythonPath -ArgumentList '-u -m final_eval.jarvis_qwen_cap' `
    -WorkingDirectory $projectRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
@{ pid = $child.Id; started_utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath $processPath -Encoding UTF8
Write-Host "Started three-prompt Qwen cap check. PID: $($child.Id)"
Write-Host "Progress: $stdoutPath"
Write-Host "Errors: $stderrPath"
Write-Host 'Maximum4 cumulative HTTP attempts, including retries. No full pilot/main starts automatically.'
