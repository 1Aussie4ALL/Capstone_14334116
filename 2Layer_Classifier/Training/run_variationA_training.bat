@echo off
echo ========================================
echo   VARIATION A CLASSIFIER TRAINING
echo ========================================
echo.
echo This will train a new classifier using:
echo - Original dataset (5,779 images)
echo - Variation A dataset (3,268 images)
echo - Total: ~9,047 images
echo.
echo Training will take several hours...
echo.
pause

echo.
echo Starting VariationA Classifier Training...
python train_variationA_classifier.py

echo.
echo Training completed!
echo Model saved as: mri_variationA_classifier.h5
echo.
pause
