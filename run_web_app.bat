@echo off
echo 🌐 Brain Tumor Classifier Web App
echo =================================
echo.
echo 🚀 Starting web application...
echo.
echo 📱 After training a model, open your browser and go to:
echo    http://localhost:5000
echo.
echo 📤 Upload MRI images to get instant Cancer/No Cancer results!
echo.
echo Press any key to start the web server...
pause > nul
echo.
python app.py
echo.
echo Web app stopped. Press any key to exit...
pause > nul
