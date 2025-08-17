@echo off
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Running photometric augmentation...
python photometric_augmentation.py

echo.
echo Press any key to exit...
pause >nul
