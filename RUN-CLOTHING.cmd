@echo off
cd /d "%~dp0"
python run_clothing.py %*
if errorlevel 1 pause
