import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify, flash
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = 'variationA-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MODEL_PATH = '../Training/mri_variationA_classifier.h5'

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to store the model
model = None
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary', 'not_mri']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_variationA_model():
    """Load the VariationA classifier model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print(f"✅ VariationA Classifier loaded successfully from: {MODEL_PATH}")
            return True
        else:
            # Try alternative paths
            alternative_paths = [
                'mri_variationA_classifier.h5',
                '../Training/mri_variationA_classifier_final.h5',
                'mri_variationA_classifier_final.h5'
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model = load_model(alt_path)
                    print(f"✅ VariationA Classifier loaded from alternative path: {alt_path}")
                    return True
            
            print(f"❌ VariationA model file not found. Tried paths:")
            print(f"   - {MODEL_PATH}")
            for path in alternative_paths:
                print(f"   - {path}")
            return False
    except Exception as e:
        print(f"❌ Error loading VariationA model: {e}")
        return False

def preprocess_image(image_path):
    """Preprocess image for VariationA prediction"""
    try:
        # Load and resize image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # VariationA model expects 128x128 images
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image_path):
    """Predict using the VariationA classifier"""
    global model
    
    if model is None:
        return None, "VariationA model not loaded"
    
    try:
        # Preprocess image
        img = preprocess_image(image_path)
        if img is None:
            return None, "Failed to preprocess image"
        
        # Make prediction
        predictions = model.predict(img)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get all class probabilities
        class_probabilities = {}
        for i, class_name in enumerate(class_names):
            class_probabilities[class_name] = float(predictions[0][i])
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'is_mri': predicted_class != 'not_mri'  # Check if predicted class is not 'not_mri'
        }, None
        
    except Exception as e:
        return None, f"VariationA prediction error: {str(e)}"

@app.route('/')
def index():
    return render_template('index_variationA_detection.html')

@app.route('/check_model')
def check_model():
    """Check if the VariationA model is loaded"""
    if model is not None:
        return jsonify({
            'status': 'success',
            'message': 'VariationA Classifier loaded successfully',
            'model_path': MODEL_PATH,
            'classes': class_names,
            'num_classes': len(class_names),
            'model_type': 'VariationA Enhanced Classifier',
            'training_info': 'Trained on original dataset + Variation A 800 photometric augmentation'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'VariationA Classifier not loaded',
            'model_path': MODEL_PATH
        })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Make prediction
            result, error = predict_image(filepath)
            
            if error:
                return jsonify({'error': error})
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Prepare response
            response = {
                'success': True,
                'filename': filename,
                'prediction': result['predicted_class'],
                'confidence': f"{result['confidence']:.2%}",
                'is_mri': result['is_mri'],
                'class_probabilities': result['class_probabilities'],
                'model_type': 'VariationA Enhanced Classifier'
            }
            
            # Add interpretation for VariationA 2-layer system
            if result['predicted_class'] == 'not_mri':
                response['interpretation'] = 'This image is NOT an MRI scan. The VariationA model detected that this is not a medical brain scan image.'
            else:
                if result['predicted_class'] == 'notumor':
                    response['interpretation'] = 'This image IS an MRI scan. The VariationA model found NO TUMOR in this brain scan. 🎉'
                else:
                    tumor_types = {
                        'glioma': 'Glioma',
                        'meningioma': 'Meningioma', 
                        'pituitary': 'Pituitary'
                    }
                    tumor_name = tumor_types.get(result['predicted_class'], result['predicted_class'])
                    response['interpretation'] = f'This image IS an MRI scan. The VariationA model detected a {tumor_name} tumor in this brain scan. ⚠️'
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'classes': class_names,
        'model_type': 'VariationA Enhanced Classifier'
    })

@app.route('/model_info')
def model_info():
    """Get detailed model information"""
    return jsonify({
        'model_name': 'VariationA Enhanced Classifier',
        'description': 'Enhanced MRI classifier trained on original dataset + Variation A 800 photometric augmentation',
        'training_data': '9,047 total images (5,779 original + 3,268 augmented)',
        'expected_accuracy': '98.5%+',
        'classes': class_names,
        'image_size': '128x128 pixels',
        'benefits': [
            '57% more training data than original classifier',
            'Enhanced data diversity through photometric augmentation',
            'Better generalization from varied examples',
            'Reduced overfitting risk',
            'Improved accuracy and robustness'
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting VariationA Enhanced MRI Detection Web App...")
    print("=" * 60)
    print("📊 Model: VariationA Enhanced Classifier")
    print("🔬 Training Data: Original + Variation A 800 augmentation")
    print("📈 Expected Performance: 98.5%+ accuracy")
    print("=" * 60)
    
    # Try to load the model
    if load_variationA_model():
        print("✅ VariationA model loaded successfully!")
        print("🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("📤 Upload any image to get enhanced MRI detection and tumor classification!")
        print("🎯 This enhanced model provides better accuracy through photometric augmentation!")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to load VariationA model. Please ensure the training has completed.")
        print("💡 Run the VariationA training first:")
        print("   cd 2Layer_Classifier/Training")
        print("   python train_variationA_classifier.py")
        print("💡 Or check if the model file exists in the Training folder.")
