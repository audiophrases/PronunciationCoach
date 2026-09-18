@echo off
setlocal
call "%~dp0_env.bat"
title Pronunciation Coach - server

where uv >nul 2>&1 || (
    echo uv is not installed. Run setup.bat first.
    pause
    exit /b 1
)

call "%~dp0_free_port.bat"
set "PC_OPEN_BROWSER=1"
echo Starting Pronunciation Coach ...
echo The browser opens by itself when the server is ready (first start takes a while: models are loading).
echo Keep this window open while you use the coach. Close it, or press Ctrl+C, to stop.
echo.
uv run python app.py
echo.
echo The server has stopped.
pause
