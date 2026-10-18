@echo off
cd /d "%~dp0"
python run_clothing.py --view-only %*
if errorlevel 1 pause
