import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """
    Load and preprocess an image for prediction
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Resize and convert to RGB
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img_normalized = img.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img, img_batch

def predict_image(image_path, model, class_names, img_size=(224, 224)):
    """
    Predict class for a single image using the MRI detection model
    """
    # Load and preprocess image
    result = load_and_preprocess_image(image_path, img_size)
    if result is None:
        return None
    
    img, img_batch = result
    
    # Make prediction
    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence, prediction[0], img

def display_prediction_result(image_path, predicted_class, confidence, probabilities, class_names, img):
    """
    Display the prediction result with the image
    """
    plt.figure(figsize=(15, 8))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Input Image: {os.path.basename(image_path)}')
    plt.axis('off')
    
    # Display prediction results
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(len(class_names)), probabilities)
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title(f'Prediction Results\nPredicted: {predicted_class} (Confidence: {confidence:.2%})')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    # Color code the bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        if i == np.argmax(probabilities):
            bar.set_color('red')
        else:
            bar.set_color('lightblue')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the trained model
    model_path = 'mri_detection_classifier.h5'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please run the training script first: python train_mri_detector.py")
        return
    
    print("Loading MRI detection model...")
    model = load_model(model_path)
    
    # Get class names from the model output shape
    num_classes = model.output_shape[-1]
    
    # Define class names (including the new 'not_mri' class)
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary', 'not_mri']
    
    print(f"Model loaded successfully!")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    while True:
        print("\n" + "="*50)
        print("MRI Detection and Tumor Classification")
        print("="*50)
        print("1. Predict single image")
        print("2. Test with sample images")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            # Get image path from user
            image_path = input("Enter the path to the image: ").strip()
            
            if not os.path.exists(image_path):
                print(f"Error: Image file '{image_path}' not found!")
                continue
            
            print(f"\nAnalyzing image: {image_path}")
            result = predict_image(image_path, model, class_names)
            
            if result:
                predicted_class, confidence, probabilities, img = result
                
                print(f"\nPrediction Results:")
                print(f"Predicted class: {predicted_class}")
                print(f"Confidence: {confidence:.2%}")
                print("\nClass probabilities:")
                for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                    print(f"  {class_name}: {prob:.2%}")
                
                # Display results
                display_prediction_result(image_path, predicted_class, confidence, probabilities, class_names, img)
                
                # Provide interpretation
                if predicted_class == 'not_mri':
                    print(f"\nInterpretation: This image is NOT an MRI scan.")
                    print("The model detected that this is not a medical brain scan image.")
                else:
                    print(f"\nInterpretation: This image IS an MRI scan.")
                    if predicted_class == 'notumor':
                        print("No tumor was detected in this brain scan.")
                    else:
                        print(f"A {predicted_class} tumor was detected in this brain scan.")
        
        elif choice == '2':
            # Test with sample images from different classes
            print("\nTesting with sample images...")
            
            # Check if we have test images available
            test_path = "Dataset/Testing"
            if os.path.exists(test_path):
                for class_name in os.listdir(test_path):
                    class_path = os.path.join(test_path, class_name)
                    if os.path.isdir(class_path):
                        # Get first image from this class
                        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if images:
                            sample_image = os.path.join(class_path, images[0])
                            print(f"\nTesting with {class_name} sample: {images[0]}")
                            
                            result = predict_image(sample_image, model, class_names)
                            if result:
                                predicted_class, confidence, probabilities, img = result
                                print(f"  Predicted: {predicted_class} (Confidence: {confidence:.2%})")
                                print(f"  Expected: {class_name}")
                                print(f"  Correct: {'Yes' if predicted_class == class_name else 'No'}")
            else:
                print("Test dataset not found. Please ensure the Dataset/Testing directory exists.")
        
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
