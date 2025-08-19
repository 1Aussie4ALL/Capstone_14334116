@echo off
echo Starting MRI Detection Training...
echo This will train a model that can distinguish between MRI and non-MRI images
echo.
python train_mri_detector.py
echo.
echo Training completed! Press any key to exit...
pause >nul
