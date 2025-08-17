import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("🚀 Starting VariationA Classifier Training (Original + Variation A Dataset)...")
print("=" * 80)

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Paths
ORIGINAL_DATASET_PATH = 'Dataset/Training'
VARIATION_A_DATASET_PATH = '../Photometric Augmentation - Variation A/Variation A_results_800'
MODEL_SAVE_PATH = 'mri_variationA_classifier.h5'

# Classes for the 2-layer system
MRI_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NON_MRI_CLASS = 'not_mri'

print(f"📁 Original dataset path: {ORIGINAL_DATASET_PATH}")
print(f"🎨 Variation A dataset path: {VARIATION_A_DATASET_PATH}")
print(f"🖼️ Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"📊 Batch size: {BATCH_SIZE}")
print(f"🔄 Epochs: {EPOCHS}")

def load_and_preprocess_data():
    """Load and preprocess both original and Variation A datasets"""
    print("\n📥 Loading combined dataset (Original + Variation A)...")
    
    images = []
    labels = []
    
    # Load original MRI images (tumor types)
    print("📁 Loading original dataset...")
    for mri_class in MRI_CLASSES:
        class_path = os.path.join(ORIGINAL_DATASET_PATH, mri_class)
        if os.path.exists(class_path):
            print(f"  Loading original {mri_class}: ", end="")
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
                        labels.append(mri_class)  # Direct tumor class
                        count += 1
            print(f"{count} images")
    
    # Load original non-MRI images
    not_mri_path = os.path.join(ORIGINAL_DATASET_PATH, 'not_mri')
    if os.path.exists(not_mri_path):
        print(f"  Loading original not_mri: ", end="")
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
    
    # Load Variation A augmented images
    print("\n🎨 Loading Variation A dataset...")
    for mri_class in MRI_CLASSES:
        class_path = os.path.join(VARIATION_A_DATASET_PATH, mri_class)
        if os.path.exists(class_path):
            print(f"  Loading Variation A {mri_class}: ", end="")
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
                        labels.append(mri_class)  # Direct tumor class
                        count += 1
            print(f"{count} images")
    
    # Load Variation A non-MRI images
    not_mri_var_path = os.path.join(VARIATION_A_DATASET_PATH, 'not_mri')
    if os.path.exists(not_mri_var_path):
        print(f"  Loading Variation A not_mri: ", end="")
        count = 0
        for filename in os.listdir(not_mri_var_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(not_mri_var_path, filename)
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

def create_variationA_model():
    """Create the VariationA classifier model"""
    print("\n🏗️ Creating VariationA model architecture...")
    
    # Base model (MobileNetV2 for efficiency)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers
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
    
    print("✅ VariationA model created successfully!")
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
    
    # Convert numeric labels back to class names for reporting
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y_val_names = [index_to_label[idx] for idx in y_val]
    y_pred_names = [index_to_label[idx] for idx in y_pred_classes]
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_val_names, y_pred_names))
    
    # Confusion matrix
    print("\n🎯 Confusion Matrix:")
    cm = confusion_matrix(y_val_names, y_pred_names, labels=list(label_to_index.keys()))
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_to_index.keys()),
                yticklabels=list(label_to_index.keys()))
    plt.title('VariationA Classifier - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_variationA.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred_classes

def plot_training_history(history):
    """Plot training history"""
    print("\n📈 Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('VariationA Classifier - Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('VariationA Classifier - Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_variationA.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("🎯 Starting VariationA Classifier Training Process...")
    print("=" * 80)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Prepare labels
    y_numeric, label_to_index = prepare_labels(y)
    
    # Create model
    model = create_variationA_model()
    
    # Train model
    history, X_val, y_val = train_model(model, X, y_numeric)
    
    # Evaluate model
    y_pred_classes = evaluate_model(model, X_val, y_val, label_to_index)
    
    # Plot training history
    plot_training_history(history)
    
    print("\n🎉 VariationA Classifier Training Completed!")
    print(f"💾 Model saved as: {MODEL_SAVE_PATH}")
    print(f"📊 Confusion matrix saved as: confusion_matrix_variationA.png")
    print(f"📈 Training history saved as: training_history_variationA.png")
    
    # Print final model performance
    final_val_accuracy = max(history.history['val_accuracy'])
    final_val_loss = min(history.history['val_loss'])
    print(f"\n🏆 Final Model Performance:")
    print(f"  Validation Accuracy: {final_val_accuracy:.4f}")
    print(f"  Validation Loss: {final_val_loss:.4f}")

if __name__ == "__main__":
    main()
