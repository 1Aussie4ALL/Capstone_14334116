@echo off
echo Adding Non-MRI Images to Dataset...
echo This will download diverse non-MRI images to improve training
echo.
python add_non_mri_images.py
echo.
echo Process completed! Press any key to exit...
pause >nul
