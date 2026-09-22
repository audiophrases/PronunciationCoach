@echo off
setlocal
call "%~dp0_env.bat"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\setup_mfa.ps1"
if errorlevel 1 echo MFA setup failed. See the error above.
pause
