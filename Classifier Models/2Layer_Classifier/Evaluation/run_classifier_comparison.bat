@echo off
echo ========================================
echo Classifier Comparison: 2Layer vs VariationA
echo ========================================
echo.
echo This will compare the performance of both
echo classifiers and generate comprehensive
echo analysis and visualizations.
echo.
echo Prerequisites:
echo - Both classifiers must be evaluated first
echo - Results should be in Evaluation_Results/
echo.
echo Press any key to continue...
pause >nul

echo.
echo 🔍 Starting classifier comparison...
echo.

python compare_classifiers.py

echo.
echo ========================================
echo Comparison completed!
echo ========================================
echo.
echo Results saved in Evaluation_Results/:
echo - classifier_comparison.csv
echo - performance_improvements.csv
echo - classifier_comparison_metrics.png
echo - performance_improvements_visualization.png
echo.
pause
