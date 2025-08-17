import time
import os
import subprocess
import psutil
import requests
from datetime import datetime, timedelta

def log_message(message):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    
    # Also save to log file (without emojis to avoid encoding issues)
    clean_message = message.encode('ascii', 'ignore').decode('ascii')
    with open("training_monitor.log", "a", encoding='utf-8') as f:
        f.write(f"[{timestamp}] {clean_message}\n")

def check_training_process():
    """Check if training process is still running"""
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        try:
            if proc.info['name'] == 'python.exe' and 'train_mri_detector_fast.py' in ' '.join(proc.cmdline()):
                return proc.info['pid'], proc.info['cpu_percent']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None, 0

def check_model_files():
    """Check which model files exist and their status"""
    files = {}
    if os.path.exists("mri_detection_classifier.h5"):
        files["final_model"] = {
            "size": os.path.getsize("mri_detection_classifier.h5"),
            "modified": datetime.fromtimestamp(os.path.getmtime("mri_detection_classifier.h5"))
        }
    
    if os.path.exists("mri_detection_classifier_best.h5"):
        files["checkpoint"] = {
            "size": os.path.getsize("mri_detection_classifier_best.h5"),
            "modified": datetime.fromtimestamp(os.path.getmtime("mri_detection_classifier_best.h5"))
        }
    
    return files

def start_training():
    """Start the training process"""
    log_message("🚀 Starting training process...")
    try:
        # Kill any existing training processes
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] == 'python.exe' and 'train_mri_detector_fast.py' in ' '.join(proc.cmdline()):
                    log_message(f"🔄 Killing existing training process {proc.info['pid']}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Start new training
        subprocess.Popen(["python", "train_mri_detector_fast.py"], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        
        log_message("✅ Training process started successfully")
        return True
    except Exception as e:
        log_message(f"❌ Failed to start training: {e}")
        return False

def test_web_app():
    """Test if the web app can load the new model"""
    log_message("🧪 Testing web app with new model...")
    
    try:
        # Start web app in background
        web_app_process = subprocess.Popen(["python", "app_mri_detection.py"], 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE)
        
        # Wait a bit for it to start
        time.sleep(10)
        
        # Test if it's responding
        try:
            response = requests.get("http://localhost:5000", timeout=5)
            if response.status_code == 200:
                log_message("✅ Web app is running and responding!")
                web_app_process.terminate()
                return True
            else:
                log_message(f"⚠️ Web app responded with status {response.status_code}")
                web_app_process.terminate()
                return False
        except requests.exceptions.RequestException:
            log_message("❌ Web app is not responding")
            web_app_process.terminate()
            return False
            
    except Exception as e:
        log_message(f"❌ Failed to test web app: {e}")
        return False

def main():
    """Main monitoring loop"""
    log_message("🔍 TRAINING MONITOR STARTED - Will run for 6+ hours")
    log_message("📋 Monitoring: Training process, model files, and web app readiness")
    
    start_time = datetime.now()
    target_time = start_time + timedelta(hours=6)
    check_interval = 30 * 60  # 30 minutes
    
    last_checkpoint_size = 0
    stuck_count = 0
    
    while datetime.now() < target_time:
        try:
            current_time = datetime.now()
            log_message(f"⏰ Check #{stuck_count + 1} - {current_time.strftime('%H:%M:%S')}")
            
            # Check training process
            pid, cpu_percent = check_training_process()
            
            if pid:
                log_message(f"✅ Training running (PID: {pid}, CPU: {cpu_percent:.1f}%)")
                
                # Check model files
                model_files = check_model_files()
                
                if "checkpoint" in model_files:
                    current_size = model_files["checkpoint"]["size"]
                    if current_size > last_checkpoint_size:
                        log_message(f"📈 Checkpoint updated: {current_size:,} bytes")
                        last_checkpoint_size = current_size
                        stuck_count = 0  # Reset stuck counter
                    else:
                        stuck_count += 1
                        log_message(f"⚠️ Checkpoint size unchanged: {current_size:,} bytes (stuck count: {stuck_count})")
                
                if "final_model" in model_files:
                    log_message("🎉 FINAL MODEL CREATED! Training completed!")
                    break
                
                # Check if training is stuck
                if stuck_count >= 3:  # 3 consecutive checks without progress
                    log_message("🚨 Training appears stuck! Restarting...")
                    if start_training():
                        stuck_count = 0
                        last_checkpoint_size = 0
                        time.sleep(60)  # Wait for restart
                        continue
                
            else:
                log_message("❌ Training process not found! Restarting...")
                if start_training():
                    stuck_count = 0
                    last_checkpoint_size = 0
                    time.sleep(60)  # Wait for restart
                    continue
            
            # Wait for next check
            log_message(f"⏳ Next check in {check_interval//60} minutes...")
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            log_message("🛑 Monitoring stopped by user")
            break
        except Exception as e:
            log_message(f"❌ Monitoring error: {e}")
            time.sleep(60)
    
    # Final verification
    log_message("🔍 FINAL VERIFICATION STARTING...")
    
    # Check final model
    if os.path.exists("mri_detection_classifier.h5"):
        final_size = os.path.getsize("mri_detection_classifier.h5")
        log_message(f"✅ FINAL MODEL READY: mri_detection_classifier.h5 ({final_size:,} bytes)")
        
        # Test web app
        if test_web_app():
            log_message("🎉 SUCCESS! Your classifier is ready and web app works!")
            log_message("🌅 You can now wake up and use your enhanced MRI detection classifier!")
        else:
            log_message("⚠️ Model ready but web app needs attention")
    else:
        log_message("❌ FINAL MODEL NOT FOUND! Attempting final restart...")
        if start_training():
            log_message("🔄 Final restart initiated - check back in 1-2 hours")
        else:
            log_message("💥 Failed to restart - manual intervention needed")
    
    log_message("🏁 MONITORING COMPLETE")

if __name__ == "__main__":
    main()
