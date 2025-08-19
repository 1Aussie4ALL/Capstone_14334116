@echo off
echo 🚀 Downloading additional non-MRI images for training...
echo.
echo This will download diverse images (nature, buildings, people, animals, etc.)
echo to improve your model's ability to distinguish MRI from non-MRI images.
echo.
echo Make sure you have internet connection!
echo.
pause
echo.
echo Starting download...
python download_non_mri_images.py
echo.
echo Download completed! Check the output above for results.
pause
