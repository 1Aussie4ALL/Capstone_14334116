@echo off
echo ========================================
echo Starting VariationA Classifier Training
echo ========================================
echo.
echo This will train a classifier using:
echo - Original dataset (Dataset/Training)
echo - Variation A 800 dataset
echo.
echo Press any key to continue...
pause >nul

echo.
echo 🚀 Starting training...
echo.

python train_variationA_classifier.py

echo.
echo ========================================
echo Training completed!
echo ========================================
echo.
echo Models saved:
echo - mri_variationA_classifier.h5 (best during training)
echo - mri_variationA_classifier_final.h5 (final model)
echo.
echo Results saved:
echo - confusion_matrix_variationA.png
echo - training_history_variationA.png
echo.
pause
