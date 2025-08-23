import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class VariationBPredictor:
    def __init__(self, model_path='../Models/mri_variationB_classifier.h5'):
        """Initialize the VariationB predictor"""
        self.model_path = model_path
        self.model = None
        self.img_size = 128
        
        # Class names
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary', 'not_mri']
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained VariationB model"""
        try:
            print(f"🔧 Loading VariationB model from: {self.model_path}")
            self.model = load_model(self.model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Make sure to train the model first!")
            return False
        return True
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not load image: {image_path}")
                return None
            
            # Resize image
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize
            img = img.astype('float32') / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict_single_image(self, image_path):
        """Predict class for a single image"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        # Preprocess image
        img = self.preprocess_image(image_path)
        if img is None:
            return None
        
        try:
            # Make prediction
            predictions = self.model.predict(img)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.classes[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx]
            
            # Get all class probabilities
            class_probabilities = {}
            for i, class_name in enumerate(self.classes):
                class_probabilities[class_name] = float(predictions[0][i])
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'all_predictions': predictions[0]
            }
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def predict_batch(self, image_folder):
        """Predict classes for all images in a folder"""
        if self.model is None:
            print("❌ Model not loaded!")
            return []
        
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Get all image files
        image_files = [f for f in os.listdir(image_folder) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        print(f"📁 Found {len(image_files)} images in {image_folder}")
        
        for i, filename in enumerate(image_files):
            image_path = os.path.join(image_folder, filename)
            print(f"🔍 Processing {i+1}/{len(image_files)}: {filename}")
            
            result = self.predict_single_image(image_path)
            if result:
                result['filename'] = filename
                result['image_path'] = image_path
                results.append(result)
                
                print(f"   ✅ Predicted: {result['predicted_class']} (Confidence: {result['confidence']:.3f})")
            else:
                print(f"   ❌ Failed to process {filename}")
        
        return results
    
    def visualize_prediction(self, image_path, prediction_result):
        """Visualize prediction results"""
        if prediction_result is None:
            print("❌ No prediction result to visualize")
            return
        
        # Load original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        ax1.imshow(img)
        ax1.set_title(f'Input Image\nPredicted: {prediction_result["predicted_class"]}')
        ax1.axis('off')
        
        # Prediction probabilities
        classes = list(prediction_result['class_probabilities'].keys())
        probabilities = list(prediction_result['class_probabilities'].values())
        
        bars = ax2.barh(classes, probabilities, color='skyblue')
        ax2.set_xlabel('Probability')
        ax2.set_title('Class Probabilities')
        ax2.set_xlim(0, 1)
        
        # Highlight predicted class
        predicted_idx = classes.index(prediction_result['predicted_class'])
        bars[predicted_idx].set_color('red')
        
        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def print_prediction_summary(self, results):
        """Print a summary of batch prediction results"""
        if not results:
            print("❌ No results to summarize")
            return
        
        print(f"\n📊 Prediction Summary")
        print("=" * 50)
        print(f"Total images processed: {len(results)}")
        
        # Count predictions per class
        class_counts = {}
        for result in results:
            pred_class = result['predicted_class']
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        
        print(f"\nPredictions per class:")
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\nAverage confidence: {avg_confidence:.3f}")

def main():
    """Main function for testing the predictor"""
    print("🚀 VariationB Classifier Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = VariationBPredictor()
    
    if predictor.model is None:
        print("❌ Cannot proceed without a trained model")
        return
    
    # Example usage
    print("\n💡 Example usage:")
    print("1. Single image prediction")
    print("2. Batch prediction from folder")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            # Single image prediction
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                result = predictor.predict_single_image(image_path)
                if result:
                    print(f"\n🎯 Prediction Result:")
                    print(f"   Class: {result['predicted_class']}")
                    print(f"   Confidence: {result['confidence']:.3f}")
                    
                    # Ask if user wants to visualize
                    viz_choice = input("Visualize prediction? (y/n): ").strip().lower()
                    if viz_choice == 'y':
                        predictor.visualize_prediction(image_path, result)
            else:
                print("❌ Image path not found!")
        
        elif choice == '2':
            # Batch prediction
            folder_path = input("Enter folder path: ").strip()
            if os.path.exists(folder_path):
                results = predictor.predict_batch(folder_path)
                if results:
                    predictor.print_prediction_summary(results)
            else:
                print("❌ Folder path not found!")
        
        elif choice == '3':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
