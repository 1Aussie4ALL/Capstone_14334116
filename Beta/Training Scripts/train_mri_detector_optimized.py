import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import seaborn as sns
import gc
import psutil

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Memory management
def clear_memory():
    """Clear memory to prevent issues"""
    gc.collect()
    tf.keras.backend.clear_session()

def check_memory():
    """Check available memory"""
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / (1024**3):.2f} GB")
    print(f"Memory usage: {memory.percent}%")
    return memory.available / (1024**3)

def load_dataset_with_mri_detection(data_path, img_size=(224, 224), max_images_per_class=1000):
    """
    Load images from the dataset directory with MRI detection
    Optimized for memory usage
    """
    X = []
    y = []
    class_names = []
    
    # Get all class directories
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            print(f"Loading {class_name} images...")
            
            # Limit images per class to prevent memory issues
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) > max_images_per_class:
                images = images[:max_images_per_class]
                print(f"  Limited to {max_images_per_class} images for memory management")
            
            # Load images
            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        X.append(img)
                        y.append(class_name)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            print(f"  Loaded {len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])} images")
    
    return np.array(X), np.array(y), class_names

def create_mri_detection_model(num_classes, img_size=(224, 224)):
    """
    Create a VGG16-based model for MRI detection and tumor classification
    """
    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', 
                       include_top=False, 
                       input_shape=(img_size[0], img_size[1], 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create the top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)  # Reduced from 1024
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)  # Reduced from 512
    x = Dropout(0.3)(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Training history plot saved as 'training_history.png'")
    except Exception as e:
        print(f"Error plotting training history: {e}")

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Confusion matrix saved as 'confusion_matrix.png'")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def main():
    print("🚀 Starting Optimized MRI Detection Training...")
    
    # Check memory before starting
    available_memory = check_memory()
    if available_memory < 4.0:  # Less than 4GB available
        print("⚠️  Warning: Low memory available. Consider closing other applications.")
    
    # Clear memory
    clear_memory()
    
    # Dataset paths
    train_path = "Dataset/Training"
    test_path = "Dataset/Testing"
    
    print("Loading training dataset...")
    X_train, y_train, class_names = load_dataset_with_mri_detection(train_path, max_images_per_class=800)
    
    print("Loading testing dataset...")
    X_test, y_test, _ = load_dataset_with_mri_detection(test_path, max_images_per_class=200)
    
    print(f"Dataset loaded successfully!")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Classes: {class_names}")
    
    # Clear memory after loading
    clear_memory()
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Convert to categorical
    num_classes = len(class_names)
    y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
    y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, num_classes)
    
    print(f"Number of classes: {num_classes}")
    
    # Create and compile model
    print("Creating model...")
    model = create_mri_detection_model(num_classes)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=15,  # Reduced from 20
        width_shift_range=0.1,  # Reduced from 0.2
        height_shift_range=0.1,  # Reduced from 0.2
        horizontal_flip=True,
        zoom_range=0.1  # Reduced from 0.2
    )
    
    # Callbacks with better error handling
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7)
    
    # Model checkpoint to save best model during training
    checkpoint = ModelCheckpoint(
        'mri_detection_classifier_checkpoint.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train the model with smaller batch size
    print("Training model...")
    try:
        history = model.fit(
            datagen.flow(X_train, y_train_categorical, batch_size=16),  # Reduced from 32
            epochs=30,  # Reduced from 50
            validation_data=(X_test, y_test_categorical),
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Evaluate model
        print("Evaluating model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Make predictions
        y_pred = model.predict(X_test, batch_size=16)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred_classes, target_names=class_names))
        
        # Plot training history
        plot_training_history(history)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test_encoded, y_pred_classes, class_names)
        
        # Save the final model
        print("Saving final model...")
        model.save('mri_detection_classifier.h5')
        print("✅ Model saved successfully as 'mri_detection_classifier.h5'")
        
        # Also save the best checkpoint
        if os.path.exists('mri_detection_classifier_checkpoint.h5'):
            print("✅ Best checkpoint model also available")
        
        print("\n🎉 Training completed successfully!")
        print("You can now use the model to classify new images.")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Attempting to save checkpoint if available...")
        
        # Try to save checkpoint if available
        if os.path.exists('mri_detection_classifier_checkpoint.h5'):
            print("✅ Checkpoint model available for use")
        else:
            print("❌ No checkpoint available. Training failed completely.")
    
    # Final memory cleanup
    clear_memory()

if __name__ == "__main__":
    main()
