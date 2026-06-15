@echo off
:: target-runner.bat for Windows

:: Set the UTF-8 environment variable inside this specific batch lifecycle
set PYTHONUTF8=1

@powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0target-runner.ps1" %*