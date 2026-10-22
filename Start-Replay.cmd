@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launchers\Start-Replay.ps1" %*
if errorlevel 1 (
  echo.
  echo Replay did not start. See the message above.
  pause
  exit /b 1
)
