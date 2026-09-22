@echo off
setlocal
call "%~dp0_env.bat"
title Pronunciation Coach - Montreal Forced Aligner

echo ============================================================
echo  Installing the Montreal Forced Aligner, which crops the words
echo  you tap. setup.bat already does this; use this if that step
echo  failed or you want to install it again.
echo  Safe to re-run: it checks what is there and only fetches the rest.
echo ============================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\setup_mfa.ps1"
if errorlevel 1 (
    echo.
    echo MFA setup failed - see the messages above.
    echo The coach still works without it, using its built-in cropper.
)
pause
