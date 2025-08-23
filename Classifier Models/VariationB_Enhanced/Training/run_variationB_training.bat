@echo off
echo ========================================
echo    VARIATION B CLASSIFIER TRAINING
echo    Original + Variation B Dataset
echo    Using 2-layer classifier as base
echo ========================================
echo.

echo Starting VariationB classifier training...
echo This will train on original + Variation B geometric dataset
echo Using 2-layer classifier as base model
echo.

python train_variationB_classifier.py

echo.
echo ========================================
echo    TRAINING COMPLETED
echo ========================================
echo.
echo Check the results:
echo - Model: mri_variationB_classifier.h5
echo - Final model: mri_variationB_classifier_final.h5
echo - Training plots: training_history_variationB.png
echo - Confusion matrix: confusion_matrix_variationB.png
echo.
pause
