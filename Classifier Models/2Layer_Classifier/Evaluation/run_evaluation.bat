@echo off
echo 🔍 Running Comprehensive 2-Layer MRI Classifier Evaluation...
echo.
echo This will generate:
echo 1. Classification reports (val + test) with per-class precision/recall/F1/support
echo 2. Normalized confusion matrices (val + test)
echo 3. PR curves with AP/PR-AUC per class
echo 4. Calibration plots with ECE + Brier scores
echo 5. Summary CSV with all metrics
echo.
echo Results will be saved in: 2Layer_Classifier/Evaluation_Results/
echo.
pause
echo.
echo Starting evaluation...
python evaluate_2layer_classifier.py
echo.
echo Evaluation completed! Check the results above and in the Evaluation_Results folder.
pause
