@echo off
setlocal
cd /d "%~dp0"
py -3 run_clothing.py --preset reference --contact-every 4
if errorlevel 1 pause
