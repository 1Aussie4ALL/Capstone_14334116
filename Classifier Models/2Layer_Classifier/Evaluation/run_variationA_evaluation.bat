@echo off
echo ========================================
echo Comprehensive VariationA Classifier Evaluation
echo ========================================
echo.
echo This will perform a comprehensive evaluation
echo of the trained VariationA classifier on both
echo training and test datasets.
echo.
echo The evaluation includes:
echo - Confusion matrices (normalized)
echo - Precision-Recall curves
echo - ROC curves  
echo - Calibration plots
echo - Comprehensive metrics
echo.
echo Press any key to continue...
pause >nul

echo.
echo 🔍 Starting comprehensive evaluation...
echo.

python evaluate_variationA_classifier.py

echo.
echo ========================================
echo Evaluation completed!
echo ========================================
echo.
echo Results saved in Evaluation_Results_VariationA/:
echo - confusion_matrix_variationA_train_normalized.png
echo - confusion_matrix_variationA_test_normalized.png
echo - pr_curves_variationA_train.png
echo - pr_curves_variationA_test.png
echo - roc_curves_variationA_train.png
echo - roc_curves_variationA_test.png
echo - calibration_variationA_train.png
echo - calibration_variationA_test.png
echo - summary_variationA_train.csv
echo - summary_variationA_test.csv
echo.
echo Compare with original 2Layer classifier results!
echo.
pause
