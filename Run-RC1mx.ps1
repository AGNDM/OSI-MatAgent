param([string]$OutputPath)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:LANGSMITH_TRACING = 'false'
$env:LANGCHAIN_TRACING_V2 = 'false'
$localConfig = Join-Path $PSScriptRoot '.env'
if (Test-Path -LiteralPath $localConfig) {
    foreach ($configLine in [System.IO.File]::ReadAllLines($localConfig)) {
        if ($configLine -match '^\s*OPENROUTER_API_KEY\s*=\s*(.+?)\s*$') {
            $env:OPENROUTER_API_KEY = $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
}
if (-not $env:OPENROUTER_API_KEY) { throw 'OPENROUTER_API_KEY is not configured.' }
if (-not $OutputPath) { $OutputPath = 'results/rc1mx-live-' + (Get-Date -Format 'yyyyMMdd-HHmmss') }
if (Test-Path -LiteralPath $OutputPath) { throw 'Output already exists; runs must not be overwritten.' }
Write-Host ('One bounded RC1mx + HFCal live-model run. Output: ' + $OutputPath)
& '.\.venv\Scripts\python.exe' -B -m app.rc1mx_experiment --output $OutputPath
if ($LASTEXITCODE -ne 0) { throw 'Runner failed. Inspect the output and logs.' }
