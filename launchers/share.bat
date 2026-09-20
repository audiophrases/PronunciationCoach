@echo off
setlocal
call "%~dp0_env.bat"
title Pronunciation Coach - shared with students

where uv >nul 2>&1 || (
    echo uv is not installed. Run setup.bat first.
    pause
    exit /b 1
)
where cloudflared >nul 2>&1 || if not exist "%ProgramFiles(x86)%\cloudflared\cloudflared.exe" (
    echo Installing cloudflared ^(approve the Windows prompt if one appears^) ...
    winget install --id Cloudflare.cloudflared --exact --accept-package-agreements --accept-source-agreements || (
        echo Could not install cloudflared.
        pause
        exit /b 1
    )
    set "PATH=%ProgramFiles(x86)%\cloudflared;%PATH%"
)

rem The session address is published through the GitHub CLI; sign in once on this machine.
where gh >nul 2>&1 || set "PATH=%ProgramFiles%\GitHub CLI;%PATH%"
gh auth status >nul 2>&1 || (
    echo Signing in to GitHub ^(needed once, to publish the coach address on the fixed page^) ...
    gh auth login --hostname github.com --git-protocol https --web || (
        echo Could not sign in to GitHub. Students can still use the session address printed below.
    )
)

rem Everything is kept for review afterwards, as with run_app_debug.bat: every assessment
rem with its per-phone detail in logs\app.log, every recording in recordings\ (wav + json),
rem and the session itself (addresses, start/stop, tunnel messages) in logs\share.log.
set "PC_DEBUG=1"
set "PC_SAVE_RECORDINGS=1"
call "%~dp0_free_port.bat"

echo Starting the coach and sharing it. Students open the fixed address shown below.
echo   session log : logs\share.log
echo   app log     : logs\app.log
echo   recordings  : recordings\
echo Keep this window open during the session. Press Ctrl+C to stop sharing.
echo.
uv run python scripts\share.py
echo.
echo Sharing has stopped. The session is in logs\share.log ^(open_logs.bat shows everything^).
pause
