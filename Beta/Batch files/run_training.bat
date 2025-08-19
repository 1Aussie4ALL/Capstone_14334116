@echo off
echo Brain Tumor Classification Training
echo =================================
echo.
echo Starting training process...
echo This may take 1-4 hours depending on your hardware
echo.
echo Press Ctrl+C to stop training at any time
echo.
pause
echo.
python train_quick.py
echo.
echo Training completed! Press any key to exit...
pause > nul
