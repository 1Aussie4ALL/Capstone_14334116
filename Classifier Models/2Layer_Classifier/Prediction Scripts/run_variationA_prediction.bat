@echo off
echo ========================================
echo VariationA Classifier - Prediction
echo ========================================
echo.
echo This will load the trained VariationA
echo classifier and allow you to make
echo predictions on new images.
echo.
echo Press any key to continue...
pause >nul

echo.
echo 🔮 Starting prediction script...
echo.

python predict_variationA.py

echo.
echo ========================================
echo Prediction completed!
echo ========================================
pause
