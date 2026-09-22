@echo off
setlocal
call "%~dp0_env.bat"
title Pronunciation Coach - compare MFA crops

where uv >nul 2>&1 || (
    echo uv is not installed. Run setup.bat first.
    pause
    exit /b 1
)

echo Realigning every archived recording in recordings\ with MFA ^(about 15 s each^) ...
echo.
uv run python scripts\mfa_compare.py "recordings/*.json" --output tmp\mfa-review.json
if errorlevel 1 (
    echo Comparison incomplete. See the errors above and tmp\mfa-review.json if it exists.
) else (
    start "" "tmp\mfa-review.html"
)
pause
