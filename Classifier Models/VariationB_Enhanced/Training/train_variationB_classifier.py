import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("🚀 Starting VariationB Classifier Training...")
print("=" * 60)
print("📊 This classifier combines original dataset + Variation B 800 dataset")
print("🔧 Uses 2-layer classifier as base model")
print("=" * 60)

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Paths
ORIGINAL_DATASET_PATH = '../../../Data/Dataset/Training'
VARIATION_B_PATH = '../../../Data/Geometric Augemntation - Variation B/Variation B_results_800'
BASE_MODEL_PATH = '../2Layer_Classifier/Models/mri_2layer_classifier_final.h5'
MODEL_SAVE_PATH = '../Models/mri_variationB_classifier.h5'

# Classes for the 2-layer system
MRI_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NON_MRI_CLASS = 'not_mri'

print(f"📁 Original dataset path: {ORIGINAL_DATASET_PATH}")
print(f"📁 Variation B dataset path: {VARIATION_B_PATH}")
print(f"🔧 Base model path: {BASE_MODEL_PATH}")
print(f"🖼️ Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"📊 Batch size: {BATCH_SIZE}")
print(f"🔄 Epochs: {EPOCHS}")

def load_and_preprocess_data():
    """Load and preprocess both original and Variation B datasets"""
    print("\n📥 Loading datasets...")
    
    images = []
    labels = []
    
    # Load original dataset
    print("  🔄 Loading original dataset...")
    for mri_class in MRI_CLASSES:
        class_path = os.path.join(ORIGINAL_DATASET_PATH, mri_class)
        if os.path.exists(class_path):
            print(f"    Loading {mri_class}: ", end="")
            count = 0
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype('float32') / 255.0
                        
                        images.append(img)
                        labels.append(mri_class)
                        count += 1
            print(f"{count} images")
    
    # Load original non-MRI images
    not_mri_path = os.path.join(ORIGINAL_DATASET_PATH, 'not_mri')
    if os.path.exists(not_mri_path):
        print(f"    Loading not_mri: ", end="")
        count = 0
        for filename in os.listdir(not_mri_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(not_mri_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype('float32') / 255.0
                    
                    images.append(img)
                    labels.append('not_mri')
                    count += 1
        print(f"{count} images")
    
    # Load Variation B dataset
    print("  🔄 Loading Variation B dataset...")
    for mri_class in MRI_CLASSES:
        class_path = os.path.join(VARIATION_B_PATH, mri_class)
        if os.path.exists(class_path):
            print(f"    Loading {mri_class} (Variation B): ", end="")
            count = 0
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype('float32') / 255.0
                        
                        images.append(img)
                        labels.append(mri_class)
                        count += 1
            print(f"{count} images")
    
    # Load Variation B non-MRI images
    not_mri_path = os.path.join(VARIATION_B_PATH, 'not_mri')
    if os.path.exists(not_mri_path):
        print(f"    Loading not_mri (Variation B): ", end="")
        count = 0
        for filename in os.listdir(not_mri_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(not_mri_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype('float32') / 255.0
                    
                    images.append(img)
                    labels.append('not_mri')
                    count += 1
        print(f"{count} images")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\n✅ Combined dataset loaded: {len(X)} total images")
    print(f"📊 Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for class_name, count in zip(unique, counts):
        print(f"  {class_name}: {count} images")
    
    return X, y

def create_variationB_model():
    """Create the VariationB classifier model using 2-layer classifier as base"""
    print("\n🏗️ Creating VariationB model architecture...")
    
    # Check if base model exists
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"⚠️ Base model not found at {BASE_MODEL_PATH}")
        print("🔄 Creating new model from scratch...")
        return create_new_model()
    
    print(f"🔧 Loading base model from: {BASE_MODEL_PATH}")
    
    try:
        # Load the pre-trained 2-layer classifier
        base_model = load_model(BASE_MODEL_PATH)
        print("✅ Base model loaded successfully!")
        
        # Create new model with same architecture but trainable
        inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        
        # Use the base model's layers but make them trainable
        x = base_model(inputs, training=True)
        
        # The base model should already have the classification layers
        # We'll use it as is for fine-tuning
        
        model = Model(inputs=inputs, outputs=x)
        
        # Make all layers trainable for fine-tuning
        for layer in model.layers:
            layer.trainable = True
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ VariationB model created successfully using base model!")
        model.summary()
        
        return model
        
    except Exception as e:
        print(f"⚠️ Error loading base model: {e}")
        print("🔄 Creating new model from scratch...")
        return create_new_model()

def create_new_model():
    """Create a new model from scratch if base model loading fails"""
    print("🏗️ Creating new model architecture...")
    
    # Base model (MobileNetV2 for efficiency)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Create the model
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer for all 5 classes
    outputs = Dense(5, activation='softmax')(x)  # 4 MRI + 1 non-MRI
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ New model created successfully!")
    model.summary()
    
    return model

def prepare_labels(y):
    """Convert string labels to numeric indices"""
    # Create label mapping
    all_classes = MRI_CLASSES + [NON_MRI_CLASS]
    label_to_index = {label: idx for idx, label in enumerate(all_classes)}
    
    # Convert labels to indices
    y_numeric = np.array([label_to_index[label] for label in y])
    
    print(f"🏷️ Label mapping:")
    for label, idx in label_to_index.items():
        print(f"  {label}: {idx}")
    
    return y_numeric, label_to_index

def train_model(model, X, y_numeric):
    """Train the model"""
    print("\n🚀 Starting training...")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
    )
    
    print(f"📊 Training set: {len(X_train)} images")
    print(f"📊 Validation set: {len(X_val)} images")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, X_val, y_val

def evaluate_model(model, X_val, y_val, label_to_index):
    """Evaluate the trained model"""
    print("\n📊 Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get class names for evaluation
    all_classes = MRI_CLASSES + [NON_MRI_CLASS]
    
    # Print classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_val, y_pred_classes, target_names=all_classes))
    
    # Create confusion matrix
    cm = confusion_matrix(y_val, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_classes, yticklabels=all_classes)
    plt.title('Confusion Matrix - VariationB Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_variationB.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_val)
    print(f"\n✅ Final Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

def main():
    """Main training function"""
    try:
        # Load data
        X, y = load_and_preprocess_data()
        
        if len(X) == 0:
            print("❌ No images found! Please check your dataset paths.")
            return
        
        # Prepare labels
        y_numeric, label_to_index = prepare_labels(y)
        
        # Create model
        model = create_variationB_model()
        
        # Train model
        history, X_val, y_val = train_model(model, X, y_numeric)
        
        # Evaluate model
        accuracy = evaluate_model(model, X_val, y_val, label_to_index)
        
        # Save final model
        model.save('mri_variationB_classifier_final.h5')
        print(f"\n💾 Model saved as: mri_variationB_classifier_final.h5")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('VariationB Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('VariationB Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history_variationB.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n🎉 VariationB training completed successfully!")
        print(f"📊 Final accuracy: {accuracy*100:.2f}%")
        print(f"💾 Best model saved: {MODEL_SAVE_PATH}")
        print(f"📈 Training plots saved: training_history_variationB.png")
        print(f"📊 Confusion matrix saved: confusion_matrix_variationB.png")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
