@echo off
setlocal
call "%~dp0_env.bat"
title Pronunciation Coach - server (DEBUG)

where uv >nul 2>&1 || (
    echo uv is not installed. Run setup.bat first.
    pause
    exit /b 1
)

rem Debug mode: full per-phone detail in the log, library output at DEBUG level,
rem and every recording archived in recordings\ (wav + json) so it can be replayed
rem by dragging the wav onto score_recording.bat.
set "PC_DEBUG=1"
set "PC_SAVE_RECORDINGS=1"
set "PC_OPEN_BROWSER=1"

echo Starting Pronunciation Coach in DEBUG mode.
echo   log file   : logs\app.log
echo   recordings : recordings\
echo The browser opens by itself when the server is ready. Close this window to stop.
echo.
uv run python app.py
echo.
echo The server has stopped. Opening the log ...
if exist "logs\app.log" start "" notepad "logs\app.log"
pause
