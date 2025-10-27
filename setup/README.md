# Unified MRI Classifier Web Application

A comprehensive multi-model deep learning web application for brain MRI tumor classification. This application simultaneously loads and runs three different CNN models trained on various augmentation strategies.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Models](#models)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

## 🎯 Overview

This web application provides a unified interface for comparing three brain MRI tumor classification models:

1. **Original 2Layer Classifier** - Baseline VGG16-based model
2. **VariationA Enhanced** - Photometric augmentation variant
3. **VariationB Enhanced** - Geometric augmentation variant

All three models run simultaneously on any uploaded MRI image, providing side-by-side comparisons with confidence scores.

## ✨ Features

- **Multi-Model Inference**: Run three models simultaneously on uploaded images
- **Real-time Predictions**: Instant tumor classification with confidence scores
- **Class Support**: Classifies 4 tumor types (glioma, meningioma, pituitary, no-tumor) + 1 non-MRI detection
- **Professional UI**: Clean, modern interface with gradient backgrounds
- **Side-by-Side Comparison**: Compare predictions across different models
- **Drag-and-Drop Upload**: Easy file upload interface
- **Confidence Visualization**: Visual representation of prediction confidence
- **Detailed Metrics**: Per-class probability distributions

## 📦 Requirements

### System Requirements

- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher (3.10+ recommended)
- **RAM**: Minimum 8GB (16GB recommended for smooth operation)
- **GPU**: Optional but recommended for faster inference

### Python Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

**Required Packages:**
- `tensorflow>=2.8.0` - Deep learning framework
- `opencv-python>=4.5.0` - Image processing
- `numpy>=1.21.0` - Numerical operations
- `Flask>=2.0.0` - Web framework
- `Werkzeug>=2.0.0` - WSGI utilities
- `Pillow>=8.3.0` - Image manipulation
- `matplotlib>=3.5.0` - Plotting
- `scikit-learn>=1.0.0` - Machine learning utilities
- `seaborn>=0.11.0` - Statistical visualization

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
# Ensure you have the complete project structure
cd "Classifier Models/WebAPP"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Navigate to parent directory
cd ..

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Navigate to setup folder
cd setup

# Run setup script
setup.bat

# Or install manually:
pip install -r requirements.txt

# Or install individually:
pip install tensorflow opencv-python numpy Flask Werkzeug Pillow matplotlib scikit-learn seaborn
```

### Step 4: Verify Model Files

Ensure all model files exist in the correct locations:

```
Classifier Models/
├── 2Layer_Classifier/
│   └── Models/
│       └── mri_2layer_classifier_final.h5 ✅
├── VariationA_Enchanced/
│   └── Models/
│       └── mri_variationA_classifier_final.h5 ✅
└── VariationB_Enhanced/
    └── Models/
        └── mri_variationB_classifier.h5 ✅
```

## 🏃 Running the Application

### Method 1: Using Batch File (Windows)

```bash
cd "Classifier Models/WebAPP"
.\run_unified_webapp.bat
```

### Method 2: Direct Python Command

```bash
cd "Classifier Models/WebAPP"
python app_unified_classifier.py
```

### Method 3: Using Flask

```bash
cd "Classifier Models/WebAPP"
flask --app app_unified_classifier run
```

**Note:** After installing dependencies from `setup` folder`, navigate back to the parent directory to run the application.

### Access the Application

Once running, open your browser and navigate to:

```
http://localhost:5000
```

The application will load all three models simultaneously and be ready for image uploads.

## 📁 Project Structure

```
Classifier Models/WebAPP/
│
├── README.md                          # This file
├── app_unified_classifier.py         # Main Flask application
├── run_unified_webapp.bat            # Windows batch launcher
│
├── templates/
│   └── index_unified_classifier.html # Web UI template
│
├── uploads/                           # User uploaded images folder
│
├── ../2Layer_Classifier/              # Model 1 directory
│   └── Models/
│       └── mri_2layer_classifier_final.h5
│
├── ../VariationA_Enchanced/           # Model 2 directory
│   └── Models/
│       └── mri_variationA_classifier_final.h5
│
└── ../VariationB_Enhanced/            # Model 3 directory
    └── Models/
        └── mri_variationB_classifier.h5
```

## 💻 Usage

### Upload and Analyze Images

1. **Launch the application** (see Running the Application)
2. **Upload an image**: Click "Choose File" or drag and drop a brain MRI image
3. **Wait for analysis**: The system will display a loading indicator
4. **View results**: See predictions from all three models with:
   - Predicted class (glioma, meningioma, pituitary, no-tumor, not_mri)
   - Confidence score (0-100%)
   - Probability distribution across all classes
5. **Upload another image**: The system automatically clears previous results

### Supported Image Formats

- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.gif`

### Image Processing

All uploaded images are:
- Automatically resized to 128x128 pixels
- Converted to RGB format
- Normalized to [0, 1] range
- Preprocessed for model compatibility

## 🤖 Models

### Model 1: Original 2Layer Classifier

- **Architecture**: VGG16 Transfer Learning
- **Training Data**: 5,779 original MRI images
- **Accuracy**: 95%+
- **Description**: Baseline classifier using frozen VGG16 backbone
- **Color**: Blue (#4a90e2)

### Model 2: VariationA Enhanced

- **Architecture**: VGG16 with Photometric Augmentation
- **Training Data**: 9,047 images (5,779 original + 3,268 augmented)
- **Accuracy**: 98.5%+
- **Augmentation**: Gamma correction, contrast, CLAHE, noise, blur, sharpen
- **Description**: Enhanced classifier with photometric data augmentation
- **Color**: Red (#e74c3c)

### Model 3: VariationB Enhanced

- **Architecture**: VGG16 with Geometric Augmentation
- **Training Data**: 9,047 images (5,779 original + 3,268 augmented)
- **Accuracy**: 97.4%+
- **Augmentation**: Rotation, translation, scaling, horizontal flip, elastic transform
- **Description**: Enhanced classifier with geometric data augmentation
- **Color**: Purple (#6c5ce7)

## 🔧 Troubleshooting

### Issue: "Model not found"

**Solution**: 
- Verify model files exist in the correct paths
- Check the console output for loading messages
- Ensure you're running from the correct directory

### Issue: "ImportError: No module named 'tensorflow'"

**Solution**:
```bash
pip install tensorflow
# Or if using GPU:
pip install tensorflow-gpu
```

### Issue: Flask server doesn't start

**Solution**:
- Check if port 5000 is already in use
- Try: `python app_unified_classifier.py --port 5001`
- Kill existing Flask processes

### Issue: Models fail to load

**Solution**:
- Verify TensorFlow/Keras versions are compatible
- Check that model files are not corrupted
- Ensure sufficient RAM (models require ~2-3GB)

### Issue: Predictions are inconsistent

**Solution**:
- Ensure images are proper brain MRI images
- Check image format compatibility
- Verify preprocessing pipeline

### Issue: Out of Memory

**Solution**:
- Reduce batch size in prediction code
- Close other memory-intensive applications
- Use CPU instead of GPU if available

## 🔬 Technical Details

### Image Preprocessing

```python
# Images are preprocessed as follows:
1. Load image using OpenCV
2. Resize to 128x128 pixels
3. Convert BGR to RGB
4. Normalize pixel values to [0, 1] range
5. Expand dimensions for batch processing
```

### Prediction Pipeline

```python
# Each model follows this pipeline:
1. Load preprocessed image
2. Pass through CNN layers
3. Apply softmax normalization
4. Select argmax for class prediction
5. Return confidence and probabilities
```

### Model Architecture

All models use transfer learning from VGG16:
- **Base**: VGG16 (ImageNet pretrained)
- **Pooling**: Global Average Pooling
- **Head**: Dense(256) → Dropout(0.3) → Dense(5) → Softmax
- **Output**: 5-class classification (4 tumor types + 1 non-MRI)

### Performance Metrics

- **Inference Time**: ~100-200ms per model (CPU)
- **Memory Usage**: ~2-3GB per model
- **Concurrent Models**: All three run simultaneously
- **Accuracy**: 95-98.5% depending on model

## 📊 API Endpoints

The application exposes several Flask endpoints:

- `GET /` - Main web interface
- `POST /upload` - Upload and process image
- `POST /analyze_current` - Analyze currently loaded image
- `POST /clear_image` - Clear current image
- `GET /health` - Health check endpoint
- `GET /get_models` - Get loaded model information
- `GET /check_models` - Check model loading status

## 🔐 Security Notes

- This is a development server (not production-ready)
- For production use, deploy with WSGI server (e.g., Gunicorn)
- Upload folder has size limits
- No authentication implemented (add for production)

## 📝 Notes for Cursor IDE

### Key Files to Understand

1. **`app_unified_classifier.py`**: Main application logic
   - Line 20-75: Model configurations
   - Line 76-120: Model loading functions
   - Line 150-220: Prediction and analysis logic
   - Line 250-280: Flask routes

2. **`index_unified_classifier.html`**: Frontend template
   - Contains all UI/UX elements
   - JavaScript for dynamic interactions
   - CSS styling and animations

### Development Environment

- **IDE**: Cursor (based on VSCode)
- **Python Version**: 3.8+ (check with `python --version`)
- **Virtual Environment**: Recommended for isolation
- **Dependencies**: Listed in `Beta/txt/requirements.txt`

### Running in Different Environments

**On a New Computer:**

1. Extract/clone the complete project
2. Navigate to `Classifier Models/WebAPP`
3. Install dependencies: `pip install -r ../../Beta/txt/requirements.txt`
4. Verify model files exist in parent directories
5. Run: `python app_unified_classifier.py`
6. Open browser to `http://localhost:5000`

### Common Commands

```bash
# Check Python version
python --version

# Install/upgrade dependencies
pip install --upgrade -r requirements.txt

# Run the application
python app_unified_classifier.py

# Check if models are loading
# Look for "SUCCESS: X models loaded successfully!" in console

# Test the application
curl http://localhost:5000/health

# Stop the application
# Press Ctrl+C in the terminal
```

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review console output for error messages
3. Verify all dependencies are installed correctly
4. Ensure model files are present and accessible

## 📄 License

This project is for academic and research purposes.

## 🎯 Summary

This unified MRI classifier web application provides a comprehensive platform for comparing three different deep learning models trained on various augmentation strategies. Simply install dependencies, ensure model files are present, and run the application to start classifying brain MRI images instantly.

---

**Quick Start Commands:**

```bash
# Navigate to app directory
cd "Classifier Models/WebAPP"

# Install dependencies
pip install -r ../../Beta/txt/requirements.txt

# Run the application
python app_unified_classifier.py

# Access in browser
# http://localhost:5000
```

---

*Last Updated: October 2025*

