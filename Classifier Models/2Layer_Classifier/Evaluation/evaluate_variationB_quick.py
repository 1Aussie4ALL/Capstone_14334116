import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, 
    average_precision_score, roc_auc_score, roc_curve
)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("🔍 Quick VariationB MRI Classifier Evaluation")
print("=" * 50)
print("📊 Fast evaluation focusing on essential metrics")
print("=" * 50)

# Configuration
IMG_SIZE = 128
MODEL_PATH = '../../VariationB_Enhanced/Models/mri_variationB_classifier.h5'
DATASET_PATH = '../../../Data/Dataset'

# Classes
MRI_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NON_MRI_CLASS = 'not_mri'
ALL_CLASSES = MRI_CLASSES + [NON_MRI_CLASS]

def load_sample_data(data_type='Testing', max_per_class=100):
    """Load a sample of the dataset for quick evaluation"""
    print(f"\n📥 Loading sample {data_type} dataset (max {max_per_class} per class)...")
    
    images = []
    labels = []
    
    # Load MRI images (tumor types)
    for mri_class in MRI_CLASSES:
        class_path = os.path.join(DATASET_PATH, data_type, mri_class)
        if os.path.exists(class_path):
            print(f"  Loading {mri_class}: ", end="")
            count = 0
            for filename in os.listdir(class_path):
                if count >= max_per_class:
                    break
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
        else:
            print(f"  {mri_class}: Directory not found")
    
    # Load non-MRI images
    non_mri_path = os.path.join(DATASET_PATH, data_type, NON_MRI_CLASS)
    if os.path.exists(non_mri_path):
        print(f"  Loading {NON_MRI_CLASS}: ", end="")
        count = 0
        for filename in os.listdir(non_mri_path):
            if count >= max_per_class:
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
    
    print(f"✅ Loaded {len(X)} sample images with {len(np.unique(y))} classes")
    return X, y

def load_model_and_predict():
    """Load the VariationB model and make predictions"""
    print(f"\n🤖 Loading VariationB model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return None, None
    
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None
    
    # Load sample test data
    X_test, y_test = load_sample_data('Testing', max_per_class=50)
    if X_test is None:
        return None, None
    
    print(f"\n🔮 Making predictions on test sample...")
    y_pred_test = model.predict(X_test, verbose=0)
    y_pred_classes_test = np.argmax(y_pred_test, axis=1)
    y_pred_test_names = [ALL_CLASSES[i] for i in y_pred_classes_test]
    
    return (X_test, y_test, y_pred_test, y_pred_test_names)

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
    """Plot ROC curves for each class"""
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(classes):
        y_binary = (y_true == class_name).astype(int)
        y_proba = y_pred_proba[:, i]
        
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_binary, y_proba)
            auc = roc_auc_score(y_binary, y_proba)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} - ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curves saved: {save_path}")

def plot_precision_recall_curves(y_true, y_pred_proba, classes, title, save_path):
    """Plot Precision-Recall curves for each class"""
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(classes):
        y_binary = (y_true == class_name).astype(int)
        y_proba = y_pred_proba[:, i]
        
        if len(np.unique(y_binary)) > 1:
            precision, recall, _ = precision_recall_curve(y_binary, y_proba)
            ap = average_precision_score(y_binary, y_proba)
            plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.3f})', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title} - Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Precision-Recall curves saved: {save_path}")

def generate_summary_report(y_true, y_pred, classes, title, save_path):
    """Generate summary report"""
    from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
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
    
    summary_df.to_csv(save_path, index=False)
    print(f"✅ Summary report saved: {save_path}")
    
    print(f"\n📊 {title} - Summary Report")
    print("=" * 50)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    print(f"\n🎯 Balanced Accuracy: {balanced_acc:.4f}")

def main():
    """Main evaluation function"""
    print("🚀 Starting Quick VariationB Classifier Evaluation...")
    
    # Create output directory
    output_dir = "Evaluation_Results_VariationB"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load model and get predictions
    test_results = load_model_and_predict()
    if test_results is None:
        print("❌ Failed to load model or data. Exiting.")
        return
    
    (X_test, y_test, y_pred_test, y_pred_test_names) = test_results
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Test sample: {len(y_test)} samples")
    
    # Generate plots and reports for test set
    print(f"\n📈 Generating test set evaluation plots...")
    plot_confusion_matrix(y_test, y_pred_test_names, ALL_CLASSES, 
                         "VariationB Test", f"{output_dir}/confusion_matrix_variationB_test_normalized.png")
    plot_roc_curves(y_test, y_pred_test, ALL_CLASSES, 
                   "VariationB Test", f"{output_dir}/roc_curves_variationB_test.png")
    plot_precision_recall_curves(y_test, y_pred_test, ALL_CLASSES, 
                                "VariationB Test", f"{output_dir}/pr_curves_variationB_test.png")
    generate_summary_report(y_test, y_pred_test_names, ALL_CLASSES, 
                           "VariationB Test", f"{output_dir}/summary_variationB_test.csv")
    
    print(f"\n🎉 Quick VariationB Classifier Evaluation Complete!")
    print(f"📁 All results saved in: {output_dir}/")
    print(f"📊 Generated {len(os.listdir(output_dir))} evaluation files")

if __name__ == "__main__":
    main()
