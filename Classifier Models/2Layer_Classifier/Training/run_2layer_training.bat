@echo off
echo 🚀 Starting 2-Layer MRI Classifier Training...
echo.
echo This will train a new model that:
echo 1. Detects if image is MRI or not
echo 2. If MRI: classifies tumor type (glioma, meningioma, notumor, pituitary)
echo 3. If not MRI: correctly identifies as non-MRI
echo.
echo Training will use your Dataset/Training folder
echo.
pause
echo.
echo Starting training...
python train_2layer_mri_classifier.py
echo.
echo Training completed! Check the output above for results.
pause
