@echo off
echo ========================================
echo Comprehensive VariationB Classifier Evaluation
echo ========================================
echo.
echo This will perform a comprehensive evaluation
echo of the trained VariationB classifier on both
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

python evaluate_variationB_classifier.py

echo.
echo ========================================
echo Evaluation completed!
echo ========================================
echo.
echo Results saved in Evaluation_Results_VariationB/:
echo - confusion_matrix_variationB_train_normalized.png
echo - confusion_matrix_variationB_test_normalized.png
echo - pr_curves_variationB_train.png
echo - pr_curves_variationB_test.png
echo - roc_curves_variationB_train.png
echo - roc_curves_variationB_test.png
echo - calibration_variationB_train.png
echo - calibration_variationB_test.png
echo - summary_variationB_train.csv
echo - summary_variationB_test.csv
echo.
echo Compare with other classifiers for analysis!
echo.
pause


