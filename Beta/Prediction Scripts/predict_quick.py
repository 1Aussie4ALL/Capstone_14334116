import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, img_size=(128, 128)):
    """
    Load and preprocess a single image for prediction
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize image
    img = cv2.resize(img, img_size)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_image(model, image_path, class_names, img_size=(128, 128)):
    """
    Predict the class of a single image
    """
    try:
        # Load and preprocess image
        img = load_and_preprocess_image(image_path, img_size)
        
        # Make prediction
        prediction = model.predict(img)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_class_idx]
        confidence = np.max(prediction)
        
        # Get all class probabilities
        probabilities = prediction[0]
        
        return predicted_class, confidence, probabilities
        
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None, None

def display_prediction_result(image_path, predicted_class, confidence, probabilities, class_names):
    """
    Display the image and prediction results
    """
    # Load image for display
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    ax1.imshow(img)
    ax1.set_title(f'Input Image\nPredicted: {predicted_class}\nConfidence: {confidence:.2%}')
    ax1.axis('off')
    
    # Display probability bar chart
    bars = ax2.bar(range(len(class_names)), probabilities)
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Probability')
    ax2.set_title('Class Probabilities')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    
    # Color the predicted class bar
    predicted_idx = class_names.index(predicted_class)
    bars[predicted_idx].set_color('red')
    
    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    # Model and class information
    model_path = 'brain_tumor_classifier_quick.h5'
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please run 'train_quick.py' first to train the model.")
        print("Or run 'run_training.bat' for easier execution.")
        return
    
    # Load the trained model
    print("Loading trained model...")
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Interactive prediction loop
    print("\nBrain Tumor Classification Model")
    print("=" * 40)
    print("Available classes:", ", ".join(class_names))
    print("Enter 'quit' to exit")
    
    while True:
        # Get image path from user
        image_path = input("\nEnter the path to an image file: ").strip()
        
        if image_path.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found!")
            continue
        
        # Check if it's an image file
        if not image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print("Error: Please provide a valid image file (.jpg, .jpeg, .png, .bmp)")
            continue
        
        # Make prediction
        print(f"\nAnalyzing image: {image_path}")
        predicted_class, confidence, probabilities = predict_image(model, image_path, class_names)
        
        if predicted_class is not None:
            print(f"\nResults:")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            print("\nAll class probabilities:")
            for class_name, prob in zip(class_names, probabilities):
                print(f"  {class_name}: {prob:.3%}")
            
            # Determine if it's cancer or no cancer
            if predicted_class == 'notumor':
                result = "NO CANCER"
                print(f"\n{'='*50}")
                print(f"🎉 FINAL RESULT: {result}")
                print(f"✅ Tumor Type: {predicted_class.upper()}")
                print(f"{'='*50}")
            else:
                result = "CANCER DETECTED"
                print(f"\n{'='*50}")
                print(f"⚠️  FINAL RESULT: {result}")
                print(f"🔴 Tumor Type: {predicted_class.upper()}")
                print(f"{'='*50}")
            
            # Display visualization
            try:
                display_prediction_result(image_path, predicted_class, confidence, probabilities, class_names)
            except Exception as e:
                print(f"Warning: Could not display visualization: {e}")
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()
