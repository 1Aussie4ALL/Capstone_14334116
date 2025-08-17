@echo off
echo Switching to V1 (Original MRI Detection)...
echo.
echo Stopping current webapp...
taskkill /f /im python.exe 2>nul
echo.
echo Copying V1 files...
copy "v1\app_mri_detection.py" "app_mri_detection.py"
copy "v1\index_mri_detection.html" "templates\index_mri_detection.html"
echo.
echo ✅ Switched to V1 successfully!
echo.
echo To start the webapp, run: python app_mri_detection.py
pause
