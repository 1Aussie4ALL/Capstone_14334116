@echo off
echo 🔍 STARTING TRAINING MONITOR
echo ================================
echo.
echo This monitor will run for 6+ hours and ensure your classifier is ready
echo It will:
echo - Check training status every 30 minutes
echo - Auto-restart if training crashes or gets stuck
echo - Verify the final model is created
echo - Test the web app with the new model
echo.
echo Expected completion: When you wake up in 6 hours
echo.
echo Press any key to start monitoring...
pause >nul

python monitor_training.py

echo.
echo Monitoring completed! Press any key to exit...
pause >nul
