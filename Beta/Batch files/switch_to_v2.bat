@echo off
echo Switching to V2 (MRI Detection Best Model)...
echo.
echo Stopping current webapp...
taskkill /f /im python.exe 2>nul
echo.
echo Copying V2 files...
copy "v2\app_mri_detection.py" "app_mri_detection.py"
copy "v2\index_mri_detection.html" "templates\index_mri_detection.html"
echo.
echo ✅ Switched to V2 successfully!
echo.
echo To start the webapp, run: python app_mri_detection.py
pause
