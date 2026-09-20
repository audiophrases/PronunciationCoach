@echo off
rem Ends a sharing session started by share.bat: the coach stops and the front door is marked closed.
cd /d "%~dp0.."
if not exist tmp mkdir tmp
echo stop > tmp\share.stop
echo Asked the sharing session to stop. The share.bat window will close the tunnel and mark the front door closed.
ping -n 4 127.0.0.1 >nul
