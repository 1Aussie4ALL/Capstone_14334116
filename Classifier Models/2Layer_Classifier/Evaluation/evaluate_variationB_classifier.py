import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, 
    average_precision_score, roc_auc_score, roc_curve, brier_score_loss,
    precision_recall_fscore_support, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("🔍 Comprehensive VariationB MRI Classifier Evaluation")
print("=" * 60)
print("📊 This classifier combines original dataset + Variation B 800 dataset")
print("=" * 60)

# Configuration
IMG_SIZE = 128
MODEL_PATH = '../../VariationB_Enhanced/Models/mri_variationB_classifier.h5'
DATASET_PATH = '../../../Data/Dataset'

# Classes
MRI_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NON_MRI_CLASS = 'not_mri'
ALL_CLASSES = MRI_CLASSES + [NON_MRI_CLASS]

def load_and_preprocess_data(data_type='Testing', max_per_class=None):
    """Load and preprocess dataset (Training or Testing)"""
    print(f"\n📥 Loading {data_type} dataset...")
    
    images = []
    labels = []
    
    # Load MRI images (tumor types)
    for mri_class in MRI_CLASSES:
        class_path = os.path.join(DATASET_PATH, data_type, mri_class)
        if os.path.exists(class_path):
            print(f"  Loading {mri_class}: ", end="")
            count = 0
            for filename in os.listdir(class_path):
                if max_per_class and count >= max_per_class:
                    break
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize to model input size
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype('float32') / 255.0
                        images.append(img)
                        labels.append(mri_class)
                        count += 1
            print(f"{count} images")
        else:
            print(f"  {mri_class}: Directory not found")
    
    # Load non-MRI images
    non_mri_path = os.path.join(DATASET_PATH, data_type, NON_MRI_CLASS)
    if os.path.exists(non_mri_path):
        print(f"  Loading {NON_MRI_CLASS}: ", end="")
        count = 0
        for filename in os.listdir(non_mri_path):
            if max_per_class and count >= max_per_class:
                break
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(non_mri_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype('float32') / 255.0
                    images.append(img)
                    labels.append(NON_MRI_CLASS)
                    count += 1
        print(f"{count} images")
    else:
        print(f"  {NON_MRI_CLASS}: Directory not found")
    
    if not images:
        print(f"❌ No images found in {data_type} dataset!")
        return None, None
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"✅ Loaded {len(X)} images with {len(np.unique(y))} classes")
    return X, y

def load_model_and_predict():
    """Load the VariationB model and make predictions"""
    print(f"\n🤖 Loading VariationB model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return None, None, None, None
    
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, None
    
    # Load test data (sample for speed)
    X_test, y_test = load_and_preprocess_data('Testing', max_per_class=100)
    if X_test is None:
        return None, None, None, None
    
    # Load train data (sample for speed)
    X_train, y_train = load_and_preprocess_data('Training', max_per_class=100)
    if X_train is None:
        return None, None, None, None
    
    print(f"\n🔮 Making predictions on test set...")
    y_pred_test = model.predict(X_test, verbose=0)
    y_pred_classes_test = np.argmax(y_pred_test, axis=1)
    
    print(f"🔮 Making predictions on train set...")
    y_pred_train = model.predict(X_train, verbose=0)
    y_pred_classes_train = np.argmax(y_pred_train, axis=1)
    
    # Convert class indices to class names
    y_pred_test_names = [ALL_CLASSES[i] for i in y_pred_classes_test]
    y_pred_train_names = [ALL_CLASSES[i] for i in y_pred_classes_train]
    
    return (X_test, y_test, y_pred_test, y_pred_test_names), (X_train, y_train, y_pred_train, y_pred_train_names)

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """Plot normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{title} - Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")

def plot_roc_curves(y_true, y_pred_proba, classes, title, save_path):
    """Plot ROC curves for each class in separate subplots"""
    print(f"\n📊 Creating ROC curves...")
    
    # Create a 2x3 grid for 5 classes
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()  # Flatten the 2D array
    
    # Calculate ROC curves for each class
    roc_aucs = []
    for i, class_name in enumerate(classes):
        # Convert to binary classification for this class
        y_binary = (y_true == class_name).astype(int)
        y_proba = y_pred_proba[:, i]
        
        if len(np.unique(y_binary)) > 1:
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_binary, y_proba)
            roc_auc = roc_auc_score(y_binary, y_proba)
            roc_aucs.append(roc_auc)
            
            # Plot
            axes[i].plot(fpr, tpr, label=f'{class_name} (AUC={roc_auc:.3f})', linewidth=2)
            axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
            axes[i].set_xlabel('False Positive Rate', fontsize=12)
            axes[i].set_ylabel('True Positive Rate', fontsize=12)
            axes[i].set_title(f'{class_name} - ROC Curve (VariationB)', fontsize=14, fontweight='bold')
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim([0, 1])
            axes[i].set_ylim([0, 1])
        else:
            roc_aucs.append(0.0)
    
    # Hide the 6th subplot (since we only have 5 classes)
    axes[5].set_visible(False)
    
    plt.suptitle('VariationB Classifier - ROC Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curves saved: {save_path}")
    
    return roc_aucs

def plot_precision_recall_curves(y_true, y_pred_proba, classes, title, save_path):
    """Plot Precision-Recall curves for each class in separate subplots"""
    print(f"\n📈 Creating Precision-Recall curves...")
    
    # Create a 2x3 grid for 5 classes
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()  # Flatten the 2D array
    
    # Calculate PR curves for each class
    pr_aucs = []
    for i, class_name in enumerate(classes):
        # Convert to binary classification for this class
        y_binary = (y_true == class_name).astype(int)
        y_proba = y_pred_proba[:, i]
        
        if len(np.unique(y_binary)) > 1:
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_binary, y_proba)
            pr_auc = average_precision_score(y_binary, y_proba)
            pr_aucs.append(pr_auc)
            
            # Plot
            axes[i].plot(recall, precision, label=f'{class_name} (AP={pr_auc:.3f})', linewidth=2)
            axes[i].set_xlabel('Recall', fontsize=12)
            axes[i].set_ylabel('Precision', fontsize=12)
            axes[i].set_title(f'{class_name} - PR Curve (VariationB)', fontsize=14, fontweight='bold')
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim([0, 1])
            axes[i].set_ylim([0, 1])
        else:
            pr_aucs.append(0.0)
    
    # Hide the 6th subplot (since we only have 5 classes)
    axes[5].set_visible(False)
    
    plt.suptitle('VariationB Classifier - Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Precision-Recall curves saved: {save_path}")
    
    return pr_aucs

def plot_calibration_curves(y_true, y_pred_proba, classes, title, save_path):
    """Plot calibration curves for each class"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, class_name in enumerate(classes):
        if i < len(axes):
            # Convert to binary classification for this class
            y_binary = (y_true == class_name).astype(int)
            y_proba = y_pred_proba[:, i]
            
            if len(np.unique(y_binary)) > 1:  # Check if class exists in data
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, y_proba, n_bins=10
                )
                
                axes[i].plot(mean_predicted_value, fraction_of_positives, "s-", 
                           label=f'{class_name}', linewidth=2, markersize=6)
                axes[i].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                axes[i].set_xlabel('Mean Predicted Probability')
                axes[i].set_ylabel('Fraction of Positives')
                axes[i].set_title(f'{class_name} Calibration')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(classes), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{title} - Calibration Curves', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Calibration curves saved: {save_path}")

def generate_summary_report(y_true, y_pred, y_pred_proba, classes, title, save_path):
    """Generate comprehensive summary report"""
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    
    # Calculate additional metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Create summary DataFrame
    summary_data = []
    for i, class_name in enumerate(classes):
        if i < len(precision):
            summary_data.append({
                'Class': class_name,
                'Precision': precision[i],
                'Recall': recall[i],
                'F1-Score': f1[i],
                'Support': support[i]
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add overall metrics
    overall_metrics = {
        'Class': 'OVERALL',
        'Precision': np.mean(precision),
        'Recall': np.mean(recall),
        'F1-Score': np.mean(f1),
        'Support': np.sum(support)
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([overall_metrics])], ignore_index=True)
    
    # Save to CSV
    summary_df.to_csv(save_path, index=False)
    print(f"✅ Summary report saved: {save_path}")
    
    # Print summary
    print(f"\n📊 {title} - Summary Report")
    print("=" * 50)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    print(f"\n🎯 Balanced Accuracy: {balanced_acc:.4f}")

def main():
    """Main evaluation function"""
    print("🚀 Starting VariationB Classifier Evaluation...")
    
    # Create output directory
    output_dir = "Evaluation_Results_VariationB"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load model and get predictions
    test_results, train_results = load_model_and_predict()
    if test_results is None or train_results is None:
        print("❌ Failed to load model or data. Exiting.")
        return
    
    (X_test, y_test, y_pred_test, y_pred_test_names) = test_results
    (X_train, y_train, y_pred_train, y_pred_train_names) = train_results
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Test set: {len(y_test)} samples")
    print(f"  Train set: {len(y_train)} samples")
    
    # Generate plots and reports for test set
    print(f"\n📈 Generating test set evaluation plots...")
    plot_confusion_matrix(y_test, y_pred_test_names, ALL_CLASSES, 
                         "VariationB Test", f"{output_dir}/confusion_matrix_variationB_test_normalized.png")
    plot_roc_curves(y_test, y_pred_test, ALL_CLASSES, 
                   "VariationB Test", f"{output_dir}/roc_curves_variationB_test.png")
    plot_precision_recall_curves(y_test, y_pred_test, ALL_CLASSES, 
                                "VariationB Test", f"{output_dir}/pr_curves_variationB_test.png")
    plot_calibration_curves(y_test, y_pred_test, ALL_CLASSES, 
                           "VariationB Test", f"{output_dir}/calibration_variationB_test.png")
    generate_summary_report(y_test, y_pred_test_names, y_pred_test, ALL_CLASSES, 
                           "VariationB Test", f"{output_dir}/summary_variationB_test.csv")
    
    # Generate plots and reports for train set
    print(f"\n📈 Generating train set evaluation plots...")
    plot_confusion_matrix(y_train, y_pred_train_names, ALL_CLASSES, 
                         "VariationB Train", f"{output_dir}/confusion_matrix_variationB_train_normalized.png")
    plot_roc_curves(y_train, y_pred_train, ALL_CLASSES, 
                   "VariationB Train", f"{output_dir}/roc_curves_variationB_train.png")
    plot_precision_recall_curves(y_train, y_pred_train, ALL_CLASSES, 
                                "VariationB Train", f"{output_dir}/pr_curves_variationB_train.png")
    plot_calibration_curves(y_train, y_pred_train, ALL_CLASSES, 
                           "VariationB Train", f"{output_dir}/calibration_variationB_train.png")
    generate_summary_report(y_train, y_pred_train_names, y_pred_train, ALL_CLASSES, 
                           "VariationB Train", f"{output_dir}/summary_variationB_train.csv")
    
    print(f"\n🎉 VariationB Classifier Evaluation Complete!")
    print(f"📁 All results saved in: {output_dir}/")
    print(f"📊 Generated {len(os.listdir(output_dir))} evaluation files")

if __name__ == "__main__":
    main()
