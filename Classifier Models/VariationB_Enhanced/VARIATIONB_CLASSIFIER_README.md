# VariationB Enhanced Classifier

## Overview
The VariationB Enhanced Classifier is a sophisticated brain tumor classification system that combines the original MRI dataset with the **Variation B (Geometric Augmentation)** dataset. It uses the **2-layer classifier** as a base model and fine-tunes it on the combined dataset.

## What It Does
- **Base Model**: Uses the pre-trained 2-layer classifier as starting point
- **Dataset**: Combines original + Variation B (geometric augmentations)
- **Training**: Fine-tunes the base model on the combined dataset
- **Output**: 5-class classifier (4 MRI tumor types + 1 non-MRI)

## Architecture
- **Input**: 128x128x3 RGB images
- **Base**: Pre-trained 2-layer classifier (transfer learning)
- **Fine-tuning**: All layers made trainable for adaptation
- **Output**: 5-class softmax classification

## Dataset Composition
- **Original Dataset**: Standard MRI training images
- **Variation B**: 800 geometrically augmented images per class
- **Total**: ~6,500+ training images
- **Classes**: glioma, meningioma, notumor, pituitary, not_mri

## Files Structure
```
VariationB_Enhanced/
├── Training/
│   ├── train_variationB_classifier.py    # Main training script
│   ├── run_variationB_training.bat       # Training batch file
│   ├── training_history_variationB.png   # Training plots
│   └── confusion_matrix_variationB.png   # Confusion matrix
├── Prediction/
│   └── predict_variationB.py             # Prediction script
├── Models/
│   ├── mri_variationB_classifier.h5      # Best model (checkpoint)
│   └── mri_variationB_classifier_final.h5 # Final trained model
└── VARIATIONB_CLASSIFIER_README.md       # This file
```

## Training Process
1. **Load Base Model**: Loads pre-trained 2-layer classifier
2. **Combine Datasets**: Merges original + Variation B data
3. **Fine-tune**: Trains on combined dataset with all layers trainable
4. **Evaluate**: Generates confusion matrix and training plots
5. **Save**: Stores best and final models

## Usage

### Training
```bash
# Run training script
python train_variationB_classifier.py

# Or use batch file (Windows)
run_variationB_training.bat
```

### Prediction
```bash
# Run prediction script
python predict_variationB.py
```

## Key Features
- ✅ **Transfer Learning**: Uses 2-layer classifier as base
- ✅ **Geometric Augmentation**: Trains on anatomically plausible variations
- ✅ **Fine-tuning**: Adapts to new data while preserving base knowledge
- ✅ **Comprehensive Evaluation**: Confusion matrix, training plots, metrics
- ✅ **Error Handling**: Graceful fallback if base model not found

## Expected Results
- **Training Plots**: Accuracy and loss curves
- **Confusion Matrix**: Classification performance visualization
- **Model Files**: Best checkpoint and final trained model
- **Performance**: Improved accuracy through geometric augmentation

## Dependencies
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Notes
- **Base Model Required**: 2-layer classifier must exist in `../2Layer_Classifier/Models/`
- **Dataset Paths**: Automatically detects and loads available data
- **Fallback**: Creates new model if base model loading fails
- **Reproducibility**: Fixed random seed for consistent results
