@echo off
setlocal
call "%~dp0_env.bat"

if exist "logs\app.log" (
    start "" notepad "logs\app.log"
) else (
    echo No log yet - run the app first. Log will appear at logs\app.log
    pause
)
if exist "logs\share.log" start "" notepad "logs\share.log"
if exist "recordings" start "" explorer "recordings"
