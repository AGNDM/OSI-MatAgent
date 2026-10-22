param(
    [string]$Task = 'cases/acceptance/v1/binder-timer.json'
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:LANGSMITH_TRACING = 'false'
$env:LANGCHAIN_TRACING_V2 = 'false'
$localConfig = Join-Path $projectRoot '.env'
if (Test-Path -LiteralPath $localConfig) {
    foreach ($configLine in [System.IO.File]::ReadAllLines($localConfig)) {
        if ($configLine -match '^\s*OPENROUTER_API_KEY\s*=\s*(.+?)\s*$') {
            $env:OPENROUTER_API_KEY = $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
}
if (-not $env:OPENROUTER_API_KEY) {
    throw 'Set OPENROUTER_API_KEY in the project .env file first.'
}
$outputPath = 'results/local-live-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
Write-Host 'Running one real OpenRouter task (paid API calls).'
Write-Host ('Results: ' + $outputPath)
& '.\.venv\Scripts\python.exe' -B -m app.instrument_cli --backend deepseek --router openrouter-jev --operation openrouter-jev --task $Task --knowledge on --output $outputPath --actions 32 --model-calls 36 --seconds 360
if ($LASTEXITCODE -ne 0) { throw 'The task process failed. Check the result directory.' }
$resultFile = Join-Path $outputPath 'result.json'
$result = Get-Content -LiteralPath $resultFile -Raw | ConvertFrom-Json
if (-not $result.score.task_success) {
    Write-Warning ('Task stopped without success: ' + $result.report.terminal_reason)
    exit 2
}
Write-Host 'The real task completed successfully.'
