@echo off
echo 🧠 Creating Challenging Augmented MRI Test Images
echo ================================================
echo.
echo This will create challenging augmented images that will really test all 3 models:
echo   • Original 2Layer Classifier
echo   • VariationA Enhanced Classifier  
echo   • VariationB Enhanced Classifier
echo.
echo The augmented images will include:
echo   • Extreme blur variations
echo   • Noise corruption
echo   • Rotation challenges
echo   • Brightness/contrast extremes
echo   • Cropping challenges
echo   • Scaling variations
echo   • Mixed augmentations
echo   • Edge cases
echo.
echo Press any key to start creating test images...
pause >nul
echo.
echo 🚀 Creating challenging test images...
python create_challenging_test_images.py
echo.
echo ✅ Test images created successfully!
echo 📁 Check the 'challenging_test_images' folder
echo 🌐 Now run the webapp and test these challenging images!
echo.
pause
