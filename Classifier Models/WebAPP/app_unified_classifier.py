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
app.secret_key = 'unified-classifier-secret-key'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Model configurations
MODELS = {
    'original': {
        'name': 'Original 2Layer Classifier',
        'path': '../2Layer_Classifier/Models/mri_2layer_classifier.h5',
        'description': 'Original MRI classifier trained on 5,779 images',
        'training_data': '5,779 images',
        'expected_accuracy': '95%+',
        'color': '#3498db'
    },
    'variationA': {
        'name': 'VariationA Enhanced Classifier',
        'path': '../VariationA_Enchanced/Models/mri_variationA_classifier.h5',
        'alternative_paths': [
            '../VariationA_Enchanced/Models/mri_variationA_classifier_final.h5',
            'mri_variationA_classifier.h5',
            'mri_variationA_classifier_final.h5'
        ],
        'description': 'Enhanced classifier with photometric augmentation',
        'training_data': '9,047 images (5,779 original + 3,268 augmented)',
        'expected_accuracy': '98.5%+',
        'color': '#e74c3c'
    }
}

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
models = {}  # Dictionary to store both models
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary', 'not_mri']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_all_models():
    """Load both classifier models simultaneously"""
    global models
    
    print("🔄 Loading both models simultaneously...")
    
    for model_key, model_config in MODELS.items():
        try:
            # Try main path first
            if os.path.exists(model_config['path']):
                models[model_key] = load_model(model_config['path'])
                print(f"✅ {model_config['name']} loaded from: {model_config['path']}")
                continue
            
            # Try alternative paths if available
            if 'alternative_paths' in model_config:
                for alt_path in model_config['alternative_paths']:
                    if os.path.exists(alt_path):
                        models[model_key] = load_model(alt_path)
                        print(f"✅ {model_config['name']} loaded from alternative path: {alt_path}")
                        break
            
            if model_key not in models:
                print(f"❌ Failed to load {model_config['name']}")
                
        except Exception as e:
            print(f"❌ Error loading {model_config['name']}: {e}")
    
    if len(models) == 2:
        print("🎉 Both models loaded successfully!")
        return True
    else:
        print(f"⚠️ Only {len(models)} out of 2 models loaded")
        return False

def preprocess_image(image_path):
    """Preprocess image for prediction (works for both models)"""
    try:
        # Load and resize image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Both models expect 128x128 images
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def predict_with_model(model, image_array, model_name):
    """Make prediction with a specific model"""
    try:
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get all class probabilities
        class_probabilities = {}
        for i, class_name in enumerate(class_names):
            class_probabilities[class_name] = float(predictions[0][i])
        
        # Determine if it's an MRI
        is_mri = predicted_class != 'not_mri'
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_mri': is_mri,
            'class_probabilities': class_probabilities,
            'model_name': model_name
        }
        
    except Exception as e:
        print(f"❌ Error predicting with {model_name}: {e}")
        return None

def predict_image(image_path):
    """Make predictions with both models"""
    try:
        # Preprocess image
        image_array = preprocess_image(image_path)
        if image_array is None:
            return None, "Failed to preprocess image"
        
        results = {}
        
        # Get predictions from both models
        for model_key, model in models.items():
            model_name = MODELS[model_key]['name']
            result = predict_with_model(model, image_array, model_name)
            if result:
                results[model_key] = result
            else:
                results[model_key] = {
                    'error': f'Failed to get prediction from {model_name}',
                    'model_name': model_name
                }
        
        return results, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

@app.route('/')
def index():
    return render_template('index_unified_classifier.html')

@app.route('/get_models')
def get_models():
    """Get available models"""
    return jsonify({
        'models': MODELS,
        'loaded_models': list(models.keys())
    })

@app.route('/check_models')
def check_models():
    """Check current models status"""
    if len(models) > 0:
        model_info = {}
        for model_key, model in models.items():
            model_config = MODELS[model_key]
            model_info[model_key] = {
                'name': model_config['name'],
                'status': 'loaded',
                'description': model_config['description'],
                'training_data': model_config['training_data'],
                'expected_accuracy': model_config['expected_accuracy']
            }
        
        return jsonify({
            'status': 'success',
            'message': f'{len(models)} models loaded successfully',
            'models': model_info,
            'classes': class_names,
            'num_classes': len(class_names)
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No models loaded',
            'models': {}
        })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction with both models"""
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
            
            # Make predictions with both models
            results, error = predict_image(filepath)
            
            if error:
                return jsonify({'error': error})
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Prepare response with both models' results
            response = {
                'success': True,
                'filename': filename,
                'results': results
            }
            
            # Add interpretations for each model
            interpretations = {}
            for model_key, result in results.items():
                if 'error' in result:
                    interpretations[model_key] = f"Error: {result['error']}"
                    continue
                
                model_name = result['model_name']
                if result['predicted_class'] == 'not_mri':
                    interpretations[model_key] = f'This image is NOT an MRI scan. The {model_name} detected that this is not a medical brain scan image.'
                else:
                    if result['predicted_class'] == 'notumor':
                        interpretations[model_key] = f'This image IS an MRI scan. The {model_name} found NO TUMOR in this brain scan. 🎉'
                    else:
                        tumor_types = {
                            'glioma': 'Glioma',
                            'meningioma': 'Meningioma', 
                            'pituitary': 'Pituitary'
                        }
                        tumor_name = tumor_types.get(result['predicted_class'], result['predicted_class'])
                        interpretations[model_key] = f'This image IS an MRI scan. The {model_name} detected a {tumor_name} tumor in this brain scan. ⚠️'
            
            response['interpretations'] = interpretations
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'loaded_models': list(models.keys()),
        'available_models': list(MODELS.keys()),
        'classes': class_names
    })

@app.route('/model_comparison')
def model_comparison():
    """Get comparison information between models"""
    return jsonify({
        'models': MODELS,
        'comparison': {
            'original_vs_variationA': {
                'data_increase': '57%',
                'accuracy_improvement': '3.5%+',
                'training_method': 'Photometric augmentation',
                'benefits': [
                    'Enhanced data diversity',
                    'Better generalization',
                    'Reduced overfitting',
                    'Improved robustness'
                ]
            }
        }
    })

if __name__ == '__main__':
    print("🚀 Starting Unified MRI Classifier Web App...")
    print("=" * 60)
    print("🔄 Multi-Model Support:")
    for key, config in MODELS.items():
        print(f"   • {config['name']}")
        print(f"     - {config['description']}")
        print(f"     - {config['training_data']}")
        print(f"     - Expected accuracy: {config['expected_accuracy']}")
    print("=" * 60)
    
    # Load both models at startup
    if load_all_models():
        print("🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🔄 Both models will provide predictions simultaneously!")
        print("📤 Upload any image to get side-by-side predictions from both classifiers!")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to load models. Please check model files and try again.")
