@echo off
rem Shared settings for every launcher. Not meant to be double-clicked.

rem Work from the project root, whatever folder the .bat was started from.
cd /d "%~dp0.."

rem UTF-8 console so IPA symbols print correctly.
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

rem --- Preferences ------------------------------------------------------
rem Target accent for the reference pronunciation: American (General American) or British.
set "PC_ACCENT=American"
rem Default first language shown in the app: Catalan, Spanish or "Other / unknown".
set "PC_L1=Catalan"
rem Whisper size for free-speech mode. base.en fits an 8 GB laptop; small.en is better but needs ~2.3 GB more.
set "PC_ASR_MODEL=base.en"
rem ---------------------------------------------------------------------

rem uv installs itself here when set up by setup.bat.
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
