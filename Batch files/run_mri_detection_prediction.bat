@echo off
echo Starting MRI Detection Prediction...
echo This will use the trained model to detect MRI images and classify tumors
echo.
python predict_mri_detection.py
echo.
echo Prediction completed! Press any key to exit...
pause >nul
