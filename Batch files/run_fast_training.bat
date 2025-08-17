@echo off
echo Starting FAST MRI Detection Training (35 epochs max)...
echo This will train your model much faster than before!
echo.
echo Expected completion time: 1-2 hours
echo.
python train_mri_detector_fast.py
echo.
echo Training completed! Press any key to exit...
pause >nul
