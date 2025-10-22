@echo off
echo 🔍 VariationB MRI Classifier Evaluation
echo ========================================
echo.
echo 📊 This will evaluate the VariationB Enhanced Classifier
echo 📈 Generates comprehensive evaluation graphs and metrics
echo.
echo 🎯 Evaluation includes:
echo   • Confusion Matrices (Normalized)
echo   • ROC Curves
echo   • Precision-Recall Curves  
echo   • Calibration Curves
echo   • Summary Reports (CSV)
echo.
echo 📁 Results will be saved in: Evaluation_Results_VariationB/
echo.
echo Press any key to start evaluation...
pause >nul
echo.
echo 🚀 Starting VariationB evaluation...
python evaluate_variationB_classifier.py
echo.
echo ✅ Evaluation complete!
echo 📁 Check the Evaluation_Results_VariationB folder for results
echo.
pause
