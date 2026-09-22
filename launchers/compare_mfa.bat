@echo off
setlocal
call "%~dp0_env.bat"
uv run python scripts\mfa_compare.py "recordings/*.json" --output tmp\mfa-review.json
if errorlevel 1 (
    echo Comparison incomplete. See the errors above and tmp\mfa-review.json if it exists.
) else (
    start "" "tmp\mfa-review.html"
)
pause
