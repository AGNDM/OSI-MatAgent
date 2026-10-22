$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$pidPath = Join-Path $projectRoot 'results/local-demo-service/pid.txt'
if (-not (Test-Path -LiteralPath $pidPath)) { Write-Host 'No managed demo process.'; exit 0 }
$demoProcessId = [int](Get-Content -LiteralPath $pidPath)
$demoProcess = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $demoProcessId)
if (-not $demoProcess) { Write-Host 'Demo already stopped.'; exit 0 }
if ($demoProcess.CommandLine -notlike ('*' + $projectRoot + '*') -or $demoProcess.CommandLine -notlike '*app.demo*') {
    throw 'Process identity changed; refusing to stop it.'
}
& taskkill.exe /PID $demoProcessId /T /F
if ($LASTEXITCODE -ne 0) { throw 'Could not stop the demo process.' }
