# Brain Tumor Classification System

This system uses deep learning to classify brain MRI images into four categories:
- **glioma** - A type of brain tumor
- **meningioma** - A type of brain tumor  
- **pituitary** - A type of brain tumor
- **notumor** - No tumor detected

## Features

- **4-class classification**: Distinguishes between different types of brain tumors and no tumor
- **Transfer learning**: Uses pre-trained VGG16 model for better accuracy
- **Data augmentation**: Improves model generalization
- **Easy prediction**: Simple script to classify new images
- **Visualization**: Shows prediction results with confidence scores

## Dataset Structure

The system expects your dataset to be organized as follows:
```
Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify your dataset structure** matches the expected format above.

## Usage

### Step 1: Train the Model

Run the training script to create your brain tumor classifier:

```bash
python train.py
```

This will:
- Load all images from your Dataset folder
- Train a VGG16-based model with transfer learning
- Save the trained model as `brain_tumor_classifier.h5`
- Display training progress and results
- Show confusion matrix and classification report

**Training time**: Depending on your hardware, this may take 1-4 hours. The script includes early stopping to prevent overfitting.

### Step 2: Classify New Images

After training, use the prediction script to classify new brain MRI images:

```bash
python predict.py
```

Then enter the path to any image file when prompted. The system will:
- Load and preprocess the image
- Make a prediction
- Display the result (Cancer/No Cancer + tumor type)
- Show confidence scores for all classes
- Provide a visual representation of the results

## Model Architecture

- **Base Model**: VGG16 pre-trained on ImageNet
- **Top Layers**: Custom dense layers with dropout for regularization
- **Input Size**: 224x224 pixels
- **Output**: 4-class softmax classification
- **Optimizer**: Adam with learning rate reduction
- **Loss Function**: Categorical crossentropy

## Training Features

- **Data Augmentation**: Rotation, shifting, flipping, and zooming
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Learning Rate Reduction**: Automatically reduces learning rate when performance plateaus
- **Dropout**: Regularization to improve generalization

## Output Interpretation

### Cancer Detection
- **"NO CANCER"**: When `notumor` class is predicted
- **"CANCER DETECTED"**: When any tumor type (glioma, meningioma, pituitary) is predicted

### Confidence Scores
- Each prediction includes confidence percentages for all classes
- Higher confidence indicates more certain predictions
- The system shows probabilities for all four classes

## Performance Metrics

The training script provides:
- Training and validation accuracy/loss curves
- Confusion matrix
- Classification report with precision, recall, and F1-score
- Final test accuracy

## Troubleshooting

### Common Issues

1. **"Model file not found"**: Run `train.py` first to create the model
2. **Memory errors**: Reduce batch size in `train.py` or use smaller images
3. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
4. **Image loading errors**: Check that image files are valid and accessible

### Performance Tips

- **GPU acceleration**: Install TensorFlow with GPU support for faster training
- **Batch size**: Adjust batch size based on available memory
- **Image size**: Reduce image size if memory is limited (modify `img_size` parameter)

## File Descriptions

- **`train.py`**: Main training script for the brain tumor classifier
- **`predict.py`**: Script to classify new images using the trained model
- **`requirements.txt`**: Python package dependencies
- **`brain_tumor_classifier.h5`**: Trained model file (created after training)

## Example Usage

```python
# After training, you can also use the model programmatically:
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('brain_tumor_classifier.h5')

# Classify a new image
# (Use the predict_image function from predict.py)
```

## Medical Disclaimer

⚠️ **Important**: This system is for educational and research purposes only. It should NOT be used for actual medical diagnosis. Always consult qualified medical professionals for medical decisions.

## Support

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Verify your dataset structure matches the expected format
3. Ensure you have sufficient memory and computational resources
4. Check the console output for specific error messages

## License

This project is provided as-is for educational purposes.
