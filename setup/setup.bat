@echo off
echo ================================================
echo Unified MRI Classifier - Setup Script
echo ================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [INFO] Python found
python --version
echo.

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not installed
    echo Please install pip
    pause
    exit /b 1
)

echo [INFO] pip found
pip --version
echo.

echo [INFO] Installing required dependencies...
echo.

REM Install dependencies from requirements file
pip install --upgrade pip
echo.

REM Check if requirements.txt exists in current setup folder
if exist "requirements.txt" (
    echo [INFO] Installing from setup/requirements.txt
    pip install -r "requirements.txt"
) else if exist "..\..\Beta\txt\requirements.txt" (
    echo [INFO] Installing from Beta/txt/requirements.txt
    pip install -r "..\..\Beta\txt\requirements.txt"
) else (
    echo [WARNING] requirements.txt not found, installing core packages...
    pip install tensorflow opencv-python numpy Flask Werkzeug Pillow matplotlib scikit-learn seaborn
)
echo.

echo [INFO] Checking for model files...
echo.

REM Check for Original 2Layer Classifier
if exist "..\..\2Layer_Classifier\Models\mri_2layer_classifier_final.h5" (
    echo [OK] Original 2Layer Classifier found
) else (
    echo [WARNING] Original 2Layer Classifier NOT found
)
echo.

REM Check for VariationA Enhanced
if exist "..\..\VariationA_Enchanced\Models\mri_variationA_classifier_final.h5" (
    echo [OK] VariationA Enhanced Classifier found
) else (
    echo [WARNING] VariationA Enhanced Classifier NOT found
)
echo.

REM Check for VariationB Enhanced
if exist "..\..\VariationB_Enhanced\Models\mri_variationB_classifier.h5" (
    echo [OK] VariationB Enhanced Classifier found
) else (
    echo [WARNING] VariationB Enhanced Classifier NOT found
)
echo.

echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo To run the application:
echo   1. Double-click run_unified_webapp.bat
echo   2. Or run: python app_unified_classifier.py
echo   3. Open browser to http://localhost:5000
echo.
pause

