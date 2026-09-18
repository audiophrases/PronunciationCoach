@echo off
rem Stop a previous Pronunciation Coach server still holding port 7860 (for example after
rem a window was closed abruptly), so the new one can start. Only python.exe is stopped;
rem anything else on that port is left alone and reported. Called by the run_app launchers.

set "PC_PORT=7860"
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%PC_PORT% .*LISTENING"') do (
    for /f "tokens=1 delims=," %%n in ('tasklist /fi "pid eq %%p" /fo csv /nh') do (
        if /i "%%~n"=="python.exe" (
            echo A previous coach server is still running ^(PID %%p^) - stopping it.
            taskkill /pid %%p /f >nul 2>&1
        ) else (
            echo Port %PC_PORT% is used by %%~n ^(PID %%p^). Close it, or the coach cannot start.
        )
    )
)
