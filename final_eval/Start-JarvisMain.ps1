param(
    [ValidateSet('gptoss','qwen')]
    [string]$Model = 'gptoss'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$outputPath = Join-Path $projectRoot 'outputs\final_eval\main_gptoss_v1\target_gptoss'
if ($Model -eq 'qwen') { $outputPath = Join-Path $projectRoot 'outputs\final_eval\main_qwen8192_v1\target_qwen' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project .venv Python is missing.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'jarvis.env'))) { throw 'Fill final_eval/jarvis.env first.' }
if (Test-Path -LiteralPath (Join-Path $outputPath 'access_check.lock')) { throw 'Main evaluation lock exists; confirm previous process stopped before restarting.' }
$processPath = Join-Path $outputPath 'process.json'
if (Test-Path -LiteralPath $processPath) {
    $previous = Get-Content -LiteralPath $processPath -Raw | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) { throw 'Previous main evaluation PID is still active.' }
}
Push-Location $projectRoot
try {
    & $pythonPath -m final_eval.jarvis_main --model $Model --dry-run
    if ($LASTEXITCODE -ne 0) { throw 'Main preflight failed; no API job launched.' }
} finally { Pop-Location }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$stdoutPath = Join-Path $outputPath 'main.stdout.log'
$stderrPath = Join-Path $outputPath 'main.stderr.log'
$child = Start-Process -FilePath $pythonPath -ArgumentList "-u -m final_eval.jarvis_main --model $Model" `
    -WorkingDirectory $projectRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
@{ pid = $child.Id; started_utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath $processPath -Encoding UTF8
Write-Host "Started main $Model accuracy evaluation. PID:$($child.Id)"
Write-Host "Progress:$stdoutPath"
Write-Host "Errors:  $stderrPath"
Write-Host '2400 logical conditions before caching; maximum3000 HTTP attempts across retries/resumptions.'
Write-Host 'Keep laptop awake and online. No local GPU/model loading; Kaggle can remain stopped.'
