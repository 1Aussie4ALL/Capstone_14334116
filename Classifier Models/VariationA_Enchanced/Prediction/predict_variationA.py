import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

print("🔮 VariationA Classifier - Prediction Script")
print("=" * 50)

# Configuration
IMG_SIZE = 128
MODEL_PATH = '../Models/mri_variationA_classifier.h5'

# Classes
MRI_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NON_MRI_CLASS = 'not_mri'
ALL_CLASSES = MRI_CLASSES + [NON_MRI_CLASS]

def load_model():
    """Load the trained VariationA model"""
    print("🤖 Loading VariationA model...")
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return model
    except:
        print(f"❌ Could not load model from {MODEL_PATH}")
        print("Trying alternative path...")
        try:
            model = tf.keras.models.load_model('mri_variationA_classifier.h5')
            print("✅ Model loaded from alternative path!")
            return model
        except:
            print("❌ Model not found. Please ensure the model file exists.")
            return None

def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    try:
        # Load and resize image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Resize to model input size
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1]
        img_normalized = img_rgb.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_resized, img_rgb
    
    except Exception as e:
        print(f"❌ Error preprocessing image: {str(e)}")
        return None

def predict_single_image(model, image_path):
    """Make prediction on a single image"""
    print(f"\n🔮 Predicting: {os.path.basename(image_path)}")
    
    # Preprocess image
    result = preprocess_image(image_path)
    if result is None:
        return None
    
    img_batch, img_resized, img_rgb = result
    
    # Make prediction
    predictions = model.predict(img_batch, verbose=0)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = ALL_CLASSES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    # Get all class probabilities
    class_probabilities = {}
    for i, class_name in enumerate(ALL_CLASSES):
        class_probabilities[class_name] = float(predictions[0][i])
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'class_probabilities': class_probabilities,
        'image_rgb': img_rgb,
        'image_resized': img_resized
    }

def display_prediction_result(image_path, result):
    """Display prediction results with visualization"""
    if result is None:
        return
    
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    class_probabilities = result['class_probabilities']
    img_rgb = result['image_rgb']
    
    print(f"\n📊 Prediction Results:")
    print(f"  🎯 Predicted Class: {predicted_class}")
    print(f"  📈 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    print(f"\n📊 All Class Probabilities:")
    for class_name, prob in sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    ax1.imshow(img_rgb)
    ax1.set_title(f'Input Image: {os.path.basename(image_path)}')
    ax1.axis('off')
    
    # Bar chart of probabilities
    classes = list(class_probabilities.keys())
    probs = list(class_probabilities.values())
    colors = ['green' if c == predicted_class else 'lightblue' for c in classes]
    
    bars = ax2.bar(classes, probs, color=colors, alpha=0.7)
    ax2.set_title('Class Probabilities')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """Main prediction function"""
    try:
        # Load model
        model = load_model()
        if model is None:
            return
        
        print("\n🎯 Ready for predictions!")
        print("=" * 40)
        
        while True:
            print("\nOptions:")
            print("1. Predict single image")
            print("2. Predict multiple images from folder")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                # Single image prediction
                image_path = input("Enter image path: ").strip()
                if os.path.exists(image_path):
                    result = predict_single_image(model, image_path)
                    display_prediction_result(image_path, result)
                else:
                    print("❌ Image file not found!")
            
            elif choice == '2':
                # Folder prediction
                folder_path = input("Enter folder path: ").strip()
                if os.path.exists(folder_path):
                    image_files = [f for f in os.listdir(folder_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    if not image_files:
                        print("❌ No image files found in folder!")
                        continue
                    
                    print(f"\n📁 Found {len(image_files)} images")
                    
                    for i, filename in enumerate(image_files[:5]):  # Limit to first 5
                        image_path = os.path.join(folder_path, filename)
                        result = predict_single_image(model, image_path)
                        if result:
                            print(f"  {filename}: {result['predicted_class']} ({result['confidence']:.2f})")
                    
                    if len(image_files) > 5:
                        print(f"  ... and {len(image_files) - 5} more images")
                
                else:
                    print("❌ Folder not found!")
            
            elif choice == '3':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
