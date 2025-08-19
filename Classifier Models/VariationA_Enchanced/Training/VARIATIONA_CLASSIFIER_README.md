# VariationA Classifier

## Overview
The **VariationA Classifier** is an enhanced version of the 2-Layer MRI Classifier that combines the original dataset with the **Variation A 800** photometric augmentation dataset. This creates a more robust classifier with increased training data diversity.

## Key Differences from Original 2Layer Classifier

### Dataset Composition
- **Original 2Layer Classifier**: Uses only the original dataset from `Dataset/Training/`
- **VariationA Classifier**: Combines:
  - Original dataset: `Dataset/Training/` (5,779 images)
  - Variation A 800 dataset: `Photometric Augmentation - Variation A/data_used_for_variation_A_800/` (3,268 images)
  - **Total**: ~9,047 images

### Expected Benefits
1. **Increased Data Diversity**: Photometric augmentations provide variations in brightness, contrast, and color
2. **Better Generalization**: More training examples help prevent overfitting
3. **Improved Robustness**: Model learns to handle different lighting and image quality conditions
4. **Higher Accuracy**: Larger training set typically leads to better performance

## Architecture
The model architecture remains the same as the original 2Layer classifier:
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 128x128x3 pixels
- **Classes**: 5 classes (4 MRI tumor types + 1 non-MRI)
- **Output**: Softmax classification

## Training Configuration
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy

## Files

### Training Script
- `train_variationA_classifier.py` - Main training script

### Batch File
- `run_variationA_training.bat` - Windows batch file to run training

### Output Files
- `mri_variationA_classifier.h5` - Best model during training
- `mri_variationA_classifier_final.h5` - Final trained model
- `confusion_matrix_variationA.png` - Confusion matrix visualization
- `training_history_variationA.png` - Training curves

## How to Train

### Option 1: Using Batch File (Windows)
```bash
run_variationA_training.bat
```

### Option 2: Direct Python Command
```bash
python train_variationA_classifier.py
```

## Dataset Structure
```
Dataset/Training/
├── glioma/          (1,317 images)
├── meningioma/      (1,336 images)
├── notumor/         (1,592 images)
├── pituitary/       (1,454 images)
└── not_mri/         (65 images)

Photometric Augmentation - Variation A/data_used_for_variation_A_800/
├── glioma/          (797 images)
├── meningioma/      (797 images)
├── notumor/         (797 images)
├── pituitary/       (797 images)
└── not_mri/         (65 images)
```

## Training Time
- **Expected Duration**: 2-4 hours (depending on hardware)
- **GPU Recommended**: Yes, for faster training
- **Memory Requirements**: ~8GB RAM minimum

## Performance Comparison
After training, you can compare the performance with the original 2Layer classifier:
- Original 2Layer: `mri_2layer_classifier.h5`
- VariationA: `mri_variationA_classifier.h5`

## Notes
- The Variation A dataset contains photometrically augmented versions of original images
- All images are resized to 128x128 pixels during training
- Data augmentation is applied through the combined dataset approach
- Early stopping prevents overfitting
- Model checkpointing saves the best performing model during training

## Troubleshooting
1. **Memory Issues**: Reduce batch size to 16 or 8
2. **CUDA Errors**: Ensure TensorFlow GPU version is installed
3. **Dataset Not Found**: Check file paths in the training script
4. **Training Stops Early**: This is normal with early stopping - check validation metrics
