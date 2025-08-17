from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for the model
model = None
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def load_model_if_exists():
    """Load the trained model if it exists"""
    global model
    model_paths = [
        'brain_tumor_classifier_instant.h5',
        'brain_tumor_classifier_quick.h5', 
        'brain_tumor_classifier.h5'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                print(f"Model loaded from: {model_path}")
                # Store model info for proper preprocessing
                if 'instant' in model_path:
                    model.model_type = 'instant'
                elif 'quick' in model_path:
                    model.model_type = 'quick'
                else:
                    model.model_type = 'full'
                return True
            except Exception as e:
                print(f"Error loading {model_path}: {e}")
                continue
    
    return False

def predict_image(image_path):
    """Predict cancer/no cancer from image"""
    global model
    
    if model is None:
        return None, None, None, "Model not loaded"
    
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None, "Could not load image"
        
        # Resize to match model input size (64x64 for instant, 128x128 for quick, 224x224 for full)
        if hasattr(model, 'model_type'):
            if model.model_type == 'instant':
                img_size = (64, 64)
            elif model.model_type == 'quick':
                img_size = (128, 128)
            else:
                img_size = (224, 224)
        else:
            # Fallback: try to detect from model name or use quick as default
            if 'instant' in str(model.name):
                img_size = (64, 64)
            elif 'quick' in str(model.name):
                img_size = (128, 128)
            else:
                img_size = (128, 128)  # Default to quick size
        
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.predict(img)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_class_idx]
        confidence = np.max(prediction)
        probabilities = prediction[0]
        
        return predicted_class, confidence, probabilities, None
        
    except Exception as e:
        return None, None, None, str(e)

@app.route('/')
def index():
    """Main page with upload form"""
    model_status = "Model loaded and ready!" if model is not None else "No model found. Please train a model first."
    return render_template('index.html', model_status=model_status)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            predicted_class, confidence, probabilities, error = predict_image(filepath)
            
            if error:
                return jsonify({'error': error})
            
            # Determine cancer/no cancer result
            if predicted_class == 'notumor':
                result = "NO CANCER"
                result_color = "success"
                icon = "✅"
            else:
                result = "CANCER DETECTED"
                result_color = "danger"
                icon = "⚠️"
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Format probabilities for display
            prob_details = []
            for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                prob_details.append({
                    'class': class_name.upper(),
                    'probability': f"{prob:.1%}",
                    'is_predicted': class_name == predicted_class
                })
            
            return jsonify({
                'success': True,
                'result': result,
                'result_color': result_color,
                'icon': icon,
                'tumor_type': predicted_class.upper(),
                'confidence': f"{confidence:.1%}",
                'probabilities': prob_details
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/check_model')
def check_model():
    """Check if model is available"""
    if model is not None:
        return jsonify({'status': 'ready', 'message': 'Model loaded and ready!'})
    else:
        return jsonify({'status': 'not_ready', 'message': 'No model found. Please train a model first.'})

if __name__ == '__main__':
    # Try to load model on startup
    if load_model_if_exists():
        print("✅ Model loaded successfully!")
    else:
        print("❌ No trained model found. Please run training first.")
        print("Available training scripts:")
        print("  - train_instant.py (few minutes)")
        print("  - train_quick.py (~15 minutes)")
        print("  - train.py (1-4 hours)")
    
    print("\n🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("📤 Upload an MRI image to get instant Cancer/No Cancer results!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
