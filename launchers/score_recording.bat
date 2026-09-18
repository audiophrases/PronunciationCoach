@echo off
setlocal
call "%~dp0_env.bat"
title Pronunciation Coach - score a recording

where uv >nul 2>&1 || (
    echo uv is not installed. Run setup.bat first.
    pause
    exit /b 1
)

if not exist tmp mkdir tmp

if "%~1"=="" (
    echo Drag a recording ^(wav, mp3, m4a, webm ...^) onto this file to score it.
    echo No file given, so scoring the built-in test clip instead.
    echo.
    uv run python scripts\smoke_test.py --plot "tmp\voices_sample.png"
    set "HEATMAP=tmp\voices_sample.png"
    goto :show
)

echo Recording: %~1
echo.
set "SENTENCE="
set /p SENTENCE="Sentence that was read (press Enter for free speech): "
echo.
if "%SENTENCE%"=="" (
    uv run python scripts\smoke_test.py --audio "%~1" --plot "tmp\%~n1.png"
) else (
    uv run python scripts\smoke_test.py --audio "%~1" --text "%SENTENCE%" --plot "tmp\%~n1.png"
)
set "HEATMAP=tmp\%~n1.png"

:show
if exist "%HEATMAP%" (
    echo.
    echo Heatmap saved as %HEATMAP% - opening it.
    start "" "%HEATMAP%"
)
echo.
pause
