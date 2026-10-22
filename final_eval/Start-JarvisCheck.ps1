param(
    [ValidateSet('deepseek', 'gptoss', 'glm', 'qwen', 'all')]
    [string]$Model = 'deepseek'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$outputPath = Join-Path $projectRoot 'outputs\final_eval\jarvis'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Project .venv Python is missing. Use the existing project Python environment first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'jarvis.env'))) {
    throw 'Copy final_eval/jarvis.env.example to final_eval/jarvis.env and fill it first.'
}
if (Test-Path -LiteralPath (Join-Path $outputPath 'access_check.lock')) {
    throw 'An API check may already be running. Check access_check.lock before starting another.'
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$stdoutPath = Join-Path $outputPath 'check.stdout.log'
$stderrPath = Join-Path $outputPath 'check.stderr.log'
$child = Start-Process -FilePath $pythonPath `
    -ArgumentList "-u -m final_eval.jarvis_api check --models $Model" `
    -WorkingDirectory $projectRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
@{ pid = $child.Id; started_ist = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow, 'India Standard Time').ToString('o'); model = $Model } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'check.process.json') -Encoding UTF8
Write-Output "Started background API check. PID: $($child.Id)"
Write-Output "Progress: $stdoutPath"
Write-Output "Errors:   $stderrPath"
Write-Output 'Two diagnostic prompts per model; at most four HTTP attempts per model across resumptions.'
