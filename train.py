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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_dataset(data_path, img_size=(224, 224)):
    """
    Load images from the dataset directory
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
            
            # Load all images in this class
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
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
    
    return np.array(X), np.array(y), class_names

def create_model(num_classes, img_size=(224, 224)):
    """
    Create a VGG16-based model for brain tumor classification
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
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
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
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def predict_single_image(model, image_path, class_names, img_size=(224, 224)):
    """
    Predict class for a single image
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence, prediction[0]

def main():
    # Dataset paths
    train_path = "Dataset/Training"
    test_path = "Dataset/Testing"
    
    print("Loading training dataset...")
    X_train, y_train, class_names = load_dataset(train_path)
    
    print("Loading testing dataset...")
    X_test, y_test, _ = load_dataset(test_path)
    
    print(f"Dataset loaded successfully!")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Classes: {class_names}")
    
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
    model = create_model(num_classes)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
    
    # Train the model
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train_categorical, batch_size=32),
        epochs=50,
        validation_data=(X_test, y_test_categorical),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred_classes, target_names=class_names))
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test_encoded, y_pred_classes, class_names)
    
    # Save the model
    model.save('brain_tumor_classifier.h5')
    print("Model saved as 'brain_tumor_classifier.h5'")
    
    # Test with a sample image
    print("\nTesting with a sample image...")
    sample_image_path = os.path.join(test_path, class_names[0], os.listdir(os.path.join(test_path, class_names[0]))[0])
    result = predict_single_image(model, sample_image_path, class_names)
    
    if result:
        predicted_class, confidence, probabilities = result
        print(f"Sample image: {sample_image_path}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print("Class probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            print(f"  {class_name}: {prob:.4f}")
    
    print("\nTraining completed! You can now use the model to classify new images.")
    print("To classify a new image, use the predict_single_image function.")

if __name__ == "__main__":
    main()
