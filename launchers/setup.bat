@echo off
setlocal
call "%~dp0_env.bat"
title Pronunciation Coach - setup

echo ============================================================
echo  Pronunciation Coach - first-time setup
echo  Installs uv, espeak-ng, ffmpeg, cloudflared and the GitHub CLI if
echo  missing, the Python packages, the Montreal Forced Aligner, and the
echo  models. About 2.5 GB is downloaded and about 6 GB used on disk.
echo ============================================================
echo.

where uv >nul 2>&1 || (
    echo [1/8] Installing uv ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" || goto :fail
)
echo [1/8] uv OK

if not exist "%ProgramFiles%\eSpeak NG\libespeak-ng.dll" (
    echo [2/8] Installing espeak-ng ... ^(approve the Windows prompt if one appears^)
    winget install --id eSpeak-NG.eSpeak-NG --exact --accept-package-agreements --accept-source-agreements || goto :fail
)
echo [2/8] espeak-ng OK

where ffmpeg >nul 2>&1 || (
    echo [3/8] Installing ffmpeg ...
    winget install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements || goto :fail
    echo       ffmpeg installed. It will be on PATH the next time you open a launcher.
)
echo [3/8] ffmpeg OK

where cloudflared >nul 2>&1 || if not exist "%ProgramFiles(x86)%\cloudflared\cloudflared.exe" (
    echo [4/8] Installing cloudflared ^(for share.bat^) ...
    winget install --id Cloudflare.cloudflared --exact --accept-package-agreements --accept-source-agreements || goto :fail
)
echo [4/8] cloudflared OK

where gh >nul 2>&1 || (
    echo [5/8] Installing the GitHub CLI ^(for share.bat^) ...
    winget install --id GitHub.cli --exact --accept-package-agreements --accept-source-agreements || goto :fail
    set "PATH=%ProgramFiles%\GitHub CLI;%PATH%"
)
echo [5/8] GitHub CLI OK

echo [6/8] Installing Python packages ...
uv sync --extra dev || goto :fail

echo [7/8] Installing the Montreal Forced Aligner ^(word cropping^) ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\setup_mfa.ps1"
if errorlevel 1 (
    echo.
    echo       MFA could not be installed - see the messages above.
    echo       Setup will continue: the coach falls back to its own cropper, and
    echo       you can finish this step later by running setup.bat again.
    echo.
)

echo [8/8] Downloading the models and scoring a test sentence ...
uv run python scripts\smoke_test.py || goto :fail

echo.
echo Setup complete. Double-click run_app.bat to use the coach here,
echo or share.bat to open it to students (it signs in to GitHub the first time).
pause
exit /b 0

:fail
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
