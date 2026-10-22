param(
    [switch]$NoBrowser,
    [switch]$CheckOnly,
    [switch]$PortablePython,
    [int]$Port = 8793
)
$ErrorActionPreference = 'Stop'
$replayRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
Set-Location -LiteralPath $replayRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$runtimeRoot = Join-Path $replayRoot '.replay-runtime/python-3.13.17'
$pythonPath = $null

function Test-ReplayPython([string]$Candidate) {
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate)) { return $false }
    try {
        & $Candidate -B -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

$portablePath = Join-Path $runtimeRoot 'python.exe'
if (Test-ReplayPython $portablePath) { $pythonPath = $portablePath }
if (-not $pythonPath -and -not $PortablePython) {
    $candidates = @((Join-Path $replayRoot '.venv/Scripts/python.exe'))
    foreach ($commandName in @('python.exe', 'python3.exe')) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($command -and $command.Source -notlike '*\WindowsApps\*') { $candidates += $command.Source }
    }
    $pyLauncher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $pyPath = & $pyLauncher.Source -3 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $pyPath) { $candidates += [string]$pyPath }
        } catch { }
    }
    foreach ($candidate in $candidates) {
        if (Test-ReplayPython $candidate) { $pythonPath = $candidate; break }
    }
}

if (-not $pythonPath) {
    # Official embeddable CPython; installed only inside this folder, without pip/admin/PATH edits.
    # Hashes: https://www.python.org/downloads/release/python-31317/
    $architecture = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) { $architecture = $env:PROCESSOR_ARCHITEW6432 }
    switch ($architecture.ToUpperInvariant()) {
        'ARM64' { $archiveArch = 'arm64'; $expectedHash = '7b58787a047b400edbfcb539aa3cdb4e2ed77bf0f22cae42b7c01cedd7dd6614' }
        'AMD64' { $archiveArch = 'amd64'; $expectedHash = '9e35319bffedab8b36ad44d8f377cb4f1733bcec257a0b6d04d94f01486ddb38' }
        'X86' { $archiveArch = 'win32'; $expectedHash = '4afd37bc88ed1391209a4c58deb902cd0741a476692412b23d340aff6e9285e7' }
        default { throw "Unsupported Windows architecture: $architecture" }
    }
    $cacheRoot = Join-Path $replayRoot '.replay-runtime'
    New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null
    $downloadPath = Join-Path $cacheRoot ('python-' + [guid]::NewGuid().ToString('N') + '.zip')
    $unpackPath = Join-Path $cacheRoot ('unpack-' + [guid]::NewGuid().ToString('N'))
    $url = "https://www.python.org/ftp/python/3.13.17/python-3.13.17-embed-$archiveArch.zip"
    Write-Host 'Preparing local Python (first launch only; about 11 MB)...'
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $downloadPath -TimeoutSec 120
        $hasher = [Security.Cryptography.SHA256]::Create()
        $downloadStream = [IO.File]::OpenRead($downloadPath)
        try { $actualHash = [BitConverter]::ToString($hasher.ComputeHash($downloadStream)).Replace('-', '').ToLowerInvariant() }
        finally { $downloadStream.Dispose(); $hasher.Dispose() }
        if ($actualHash -ne $expectedHash) {
            throw 'Python download checksum mismatch. Please retry.'
        }
        # Use .NET directly: Windows PowerShell may inherit a PSModulePath from
        # PowerShell 7 that prevents Get-FileHash / Expand-Archive autoloading.
        [void][Reflection.Assembly]::LoadWithPartialName('System.IO.Compression.FileSystem')
        [IO.Compression.ZipFile]::ExtractToDirectory($downloadPath, $unpackPath)
        if (-not (Test-ReplayPython (Join-Path $unpackPath 'python.exe'))) {
            throw 'The downloaded Python cannot run on this Windows installation.'
        }
        if (Test-Path -LiteralPath $runtimeRoot) {
            # Keep a damaged/older cache recoverable rather than deleting it.
            $backupPath = $runtimeRoot + '.previous-' + [guid]::NewGuid().ToString('N')
            foreach ($movePath in @($runtimeRoot, $backupPath)) {
                if (-not [IO.Path]::GetFullPath($movePath).StartsWith($cacheRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
                    throw 'Runtime cache path is outside the project.'
                }
            }
            Move-Item -LiteralPath $runtimeRoot -Destination $backupPath
        }
        foreach ($movePath in @($unpackPath, $runtimeRoot)) {
            if (-not [IO.Path]::GetFullPath($movePath).StartsWith($cacheRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Runtime cache path is outside the project.'
            }
        }
        Move-Item -LiteralPath $unpackPath -Destination $runtimeRoot
        $pythonPath = $portablePath
    } finally {
        if (Test-Path -LiteralPath $downloadPath) { Remove-Item -LiteralPath $downloadPath }
    }
}

Write-Host 'Starting the saved instrument replay...'
$launchArguments = @('-B', (Join-Path $replayRoot 'tools/launch_replay.py'), '--port', [string]$Port)
if ($NoBrowser) { $launchArguments += '--no-browser' }
if ($CheckOnly) { $launchArguments += '--check' }
& $pythonPath @launchArguments
exit $LASTEXITCODE
