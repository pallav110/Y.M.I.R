@echo off
echo Starting Y.M.I.R AI Emotion Detection System...
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Run the launcher
python start_ymir.py

pause