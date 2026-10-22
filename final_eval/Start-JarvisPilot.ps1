param(
    [ValidateSet('deepseek', 'gptoss', 'qwen')]
    [string]$Model = 'deepseek',
    [ValidateSet(4096, 8192)]
    [int]$QwenOutputCap = 4096
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$readerFolder = $Model
if ($Model -eq 'qwen' -and $QwenOutputCap -eq 8192) { $readerFolder = 'qwen_8192' }
$outputPath = Join-Path $projectRoot "outputs\final_eval\target_pilots\$readerFolder"
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project .venv Python is missing.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'jarvis.env'))) { throw 'Fill final_eval/jarvis.env first.' }
if (Test-Path -LiteralPath (Join-Path $outputPath 'access_check.lock')) {
    throw 'Pilot lock exists. Confirm the previous process stopped before removing a stale lock.'
}
$processPath = Join-Path $outputPath 'process.json'
if (Test-Path -LiteralPath $processPath) {
    $previous = Get-Content -LiteralPath $processPath -Raw | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) { throw 'Previous pilot PID is still active.' }
}
Push-Location $projectRoot
try {
    & $pythonPath -m final_eval.jarvis_pilot --model $Model --qwen-output-cap $QwenOutputCap --dry-run
    if ($LASTEXITCODE -ne 0) { throw 'Pilot preflight failed; nothing launched.' }
} finally { Pop-Location }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$stdoutPath = Join-Path $outputPath 'pilot.stdout.log'
$stderrPath = Join-Path $outputPath 'pilot.stderr.log'
$child = Start-Process -FilePath $pythonPath -ArgumentList "-u -m final_eval.jarvis_pilot --model $Model --qwen-output-cap $QwenOutputCap" `
    -WorkingDirectory $projectRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
@{ pid = $child.Id; started_utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath $processPath -Encoding UTF8
Write-Host "Started $Model target pilot. PID: $($child.Id)"
Write-Host "Progress: $stdoutPath"
Write-Host "Errors:   $stderrPath"
Write-Host 'Keep the laptop awake and connected. Closing this launcher does not stop the background process.'
