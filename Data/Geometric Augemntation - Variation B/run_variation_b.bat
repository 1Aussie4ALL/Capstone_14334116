@echo off
echo ========================================
echo    VARIATION B DATASET CREATION
echo    Geometric Augmentation (800 per class)
echo ========================================
echo.

echo Starting Variation B geometric augmentation...
echo This will create anatomically plausible geometric transformations
echo.

python create_variation_b_dataset_800.py

echo.
echo ========================================
echo    PROCESS COMPLETED
echo ========================================
echo.
echo Check the results in: Variation B_results_800/
echo Summary report: Variation_B_800_Summary_Report.txt
echo.
pause
