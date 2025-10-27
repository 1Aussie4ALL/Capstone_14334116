# Quick Start Guide - Unified MRI Classifier

Get up and running in 5 minutes! 🚀

## ⚡ Fast Setup (Windows)

### Step 1: Install Python Dependencies

Navigate to setup folder and run:
```bash
cd setup
setup.bat
```

Or manually install:
```bash
cd setup
pip install -r requirements.txt
```

### Step 2: Start the Application

**Option A: Batch File**
```bash
run_unified_webapp.bat
```

**Option B: Python Command**
```bash
python app_unified_classifier.py
```

### Step 3: Open Browser

Navigate to: **http://localhost:5000**

## 📝 What You Need

1. **Python 3.8+** installed
2. **Model files** in correct directories
3. **Internet connection** (for first-time dependency download)

## ✅ Verification Checklist

Before running, verify:

- [ ] Python installed (`python --version`)
- [ ] All dependencies installed (`pip list`)
- [ ] Model files exist in parent directories
- [ ] Port 5000 is available
- [ ] Uploads folder exists (auto-created)

## 🎯 Quick Test

1. Run the app
2. Upload any brain MRI image
3. Wait for analysis (10-30 seconds)
4. View results from all 3 models

## 🐛 Common Issues

### "ModuleNotFoundError"

**Fix:**
```bash
pip install <module-name>
# Or install all at once:
pip install -r requirements.txt
```

### "Model not found"

**Fix:**
- Ensure model files exist in `../` directories
- Check console output for exact path errors

### "Port already in use"

**Fix:**
```bash
# Kill the process using port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### "Memory Error"

**Fix:**
- Close other applications
- Use smaller images
- Increase virtual memory

## 📱 Usage

1. **Launch**: Run `python app_unified_classifier.py`
2. **Upload**: Choose a brain MRI image
3. **Analyze**: Click "Analyze Image"
4. **Compare**: View predictions from 3 models
5. **Repeat**: Upload another image

## 🎓 Example

```
> python app_unified_classifier.py

Starting Unified MRI Classifier Web App...
SUCCESS: 3 models loaded successfully!
Running on http://localhost:5000

[Open browser and upload an image]
```

That's it! You're ready to classify brain MRI images! 🎉

---

**Need help?** Check `README.md` for detailed documentation.

