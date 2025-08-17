@echo off
echo Switching to V3 (Brain Tumor Classifier Quick)...
echo.
echo Stopping current webapp...
taskkill /f /im python.exe 2>nul
echo.
echo Copying V3 files...
copy "v3\app_mri_detection.py" "app_mri_detection.py"
copy "v3\index_mri_detection.html" "templates\index_mri_detection.html"
echo.
echo ✅ Switched to V3 successfully!
echo.
echo To start the webapp, run: python app_mri_detection.py
pause
