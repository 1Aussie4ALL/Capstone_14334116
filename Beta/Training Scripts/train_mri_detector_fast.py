import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_dataset_fast(data_path, img_size=(224, 224), max_images_per_class=600):
    """
    Load images efficiently with memory management
    """
    X = []
    y = []
    class_names = []
    
    print("Loading dataset efficiently...")
    
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            print(f"Loading {class_name}...")
            
            # Get image files
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit images per class for faster training
            if len(images) > max_images_per_class:
                images = images[:max_images_per_class]
                print(f"  Limited to {max_images_per_class} images for speed")
            
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
            
            print(f"  Loaded {len(images)} images")
    
    return np.array(X), np.array(y), class_names

def create_fast_model(num_classes, img_size=(224, 224)):
    """
    Create a streamlined VGG16 model for fast training
    """
    print("Creating streamlined model...")
    
    # Load pre-trained VGG16
    base_model = VGG16(weights='imagenet', 
                       include_top=False, 
                       input_shape=(img_size[0], img_size[1], 3))
    
    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create top layers (simplified for speed)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

def plot_results(history, y_true, y_pred, class_names):
    """
    Plot training results efficiently
    """
    try:
        # Training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        print("✅ Training results saved as 'training_results.png'")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("✅ Confusion matrix saved as 'confusion_matrix.png'")
        
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")

def main():
    print("🚀 Starting FAST MRI Detection Training (35 epochs max)")
    print("=" * 60)
    
    # Dataset paths
    train_path = "Dataset/Training"
    test_path = "Dataset/Testing"
    
    # Load datasets efficiently
    print("\n📁 Loading training dataset...")
    X_train, y_train, class_names = load_dataset_fast(train_path, max_images_per_class=600)
    
    print("\n📁 Loading testing dataset...")
    X_test, y_test, _ = load_dataset_fast(test_path, max_images_per_class=150)
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Classes: {class_names}")
    
    # Normalize and prepare data
    print("\n⚙️  Preparing data...")
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    num_classes = len(class_names)
    y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
    y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, num_classes)
    
    print(f"Number of classes: {num_classes}")
    
    # Create and compile model
    print("\n🏗️  Creating model...")
    model = create_fast_model(num_classes)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())
    
    # Data augmentation (minimal for speed)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # Callbacks for fast training
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=6, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.3, 
        patience=3, 
        min_lr=1e-6,
        verbose=1
    )
    
    # Save best model during training
    checkpoint = ModelCheckpoint(
        'mri_detection_classifier_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train the model
    print("\n🚀 Starting training (35 epochs max)...")
    print("Expected completion: 1-2 hours")
    
    try:
        history = model.fit(
            datagen.flow(X_train, y_train_categorical, batch_size=24),
            epochs=35,  # Maximum 35 epochs
            validation_data=(X_test, y_test_categorical),
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        print("\n✅ Training completed successfully!")
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}")
        
        # Make predictions
        print("\n🔮 Making predictions...")
        y_pred = model.predict(X_test, batch_size=24)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Print results
        print("\n📋 Classification Report:")
        print(classification_report(y_test_encoded, y_pred_classes, target_names=class_names))
        
        # Plot results
        print("\n📈 Creating visualizations...")
        plot_results(history, y_test_encoded, y_pred_classes, class_names)
        
        # Save final model
        print("\n💾 Saving final model...")
        model.save('mri_detection_classifier.h5')
        print("✅ Final model saved as 'mri_detection_classifier.h5'")
        
        # Check for best checkpoint
        if os.path.exists('mri_detection_classifier_best.h5'):
            print("✅ Best checkpoint model also available")
        
        print("\n🎉 SUCCESS! Training completed in", len(history.history['accuracy']), "epochs")
        print("Your enhanced MRI detection classifier is ready!")
        print("\nNext steps:")
        print("1. Run: python app_mri_detection.py")
        print("2. Test with any image (MRI or non-MRI)")
        print("3. Enjoy accurate classification!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Attempting to save checkpoint if available...")
        
        if os.path.exists('mri_detection_classifier_best.h5'):
            print("✅ Checkpoint model available for use")
            print("You can still use the partially trained model!")
        else:
            print("❌ No checkpoint available. Training failed.")
            print("Please check your system resources and try again.")

if __name__ == "__main__":
    main()
