param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:LANGSMITH_TRACING = 'false'
$env:LANGCHAIN_TRACING_V2 = 'false'
$demoUrl = 'http://127.0.0.1:8793'
$demoLogs = Join-Path $projectRoot 'results/local-demo-service'
New-Item -ItemType Directory -Path $demoLogs -Force | Out-Null
$ready = $false
try {
    $state = Invoke-RestMethod -Uri ($demoUrl + '/api/state') -TimeoutSec 2
    $ready = [bool]$state.modes.'scripted-rules'
} catch { }
if (-not $ready) {
    $demoProcess = Start-Process -FilePath (Join-Path $projectRoot '.venv/Scripts/python.exe') -ArgumentList @('-B','-m','app.demo','--port','8793','--output','results/local-drying-demo') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $demoLogs 'stdout.log') -RedirectStandardError (Join-Path $demoLogs 'stderr.log') -PassThru
    $demoProcess.Id | Set-Content -LiteralPath (Join-Path $demoLogs 'pid.txt')
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $state = Invoke-RestMethod -Uri ($demoUrl + '/api/state') -TimeoutSec 2
            $ready = [bool]$state.modes.'scripted-rules'
            if ($ready) { break }
        } catch { }
        if ($demoProcess.HasExited) { break }
    }
}
if (-not $ready) { throw ('Demo failed to start. See ' + $demoLogs) }
Write-Host ('Workflow result viewer: ' + $demoUrl)
if (-not $NoBrowser) { Start-Process $demoUrl }
