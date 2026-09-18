@echo off
setlocal
call "%~dp0_env.bat"
title Pronunciation Coach - setup

echo ============================================================
echo  Pronunciation Coach - first-time setup
echo  Installs uv, espeak-ng and ffmpeg if missing, the Python
echo  packages, and downloads the models (about 1.5 GB).
echo ============================================================
echo.

where uv >nul 2>&1 || (
    echo [1/5] Installing uv ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" || goto :fail
)
echo [1/5] uv OK

if not exist "%ProgramFiles%\eSpeak NG\libespeak-ng.dll" (
    echo [2/5] Installing espeak-ng ... ^(approve the Windows prompt if one appears^)
    winget install --id eSpeak-NG.eSpeak-NG --exact --accept-package-agreements --accept-source-agreements || goto :fail
)
echo [2/5] espeak-ng OK

where ffmpeg >nul 2>&1 || (
    echo [3/5] Installing ffmpeg ...
    winget install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements || goto :fail
    echo       ffmpeg installed. It will be on PATH the next time you open a launcher.
)
echo [3/5] ffmpeg OK

echo [4/5] Installing Python packages ...
uv sync || goto :fail

echo [5/5] Downloading the models and scoring a test sentence ...
uv run python scripts\smoke_test.py || goto :fail

echo.
echo Setup complete. Double-click run_app.bat to start the coach.
pause
exit /b 0

:fail
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
