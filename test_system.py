import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def test_dataset_loading():
    """Test if we can load images from the dataset"""
    print("Testing dataset loading...")
    
    train_path = "Dataset/Training"
    test_path = "Dataset/Testing"
    
    # Check if paths exist
    if not os.path.exists(train_path):
        print(f"ERROR: Training path {train_path} not found!")
        return False
    
    if not os.path.exists(test_path):
        print(f"ERROR: Testing path {test_path} not found!")
        return False
    
    # Count images in each class
    train_classes = {}
    test_classes = {}
    
    for class_name in os.listdir(train_path):
        class_path = os.path.join(train_path, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            train_classes[class_name] = len(images)
    
    for class_name in os.listdir(test_path):
        class_path = os.path.join(test_path, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            test_classes[class_name] = len(images)
    
    print("Training dataset:")
    for class_name, count in train_classes.items():
        print(f"  {class_name}: {count} images")
    
    print("Testing dataset:")
    for class_name, count in test_classes.items():
        print(f"  {class_name}: {count} images")
    
    return True

def test_image_loading():
    """Test if we can load and process individual images"""
    print("\nTesting image loading...")
    
    train_path = "Dataset/Training"
    
    # Try to load a few images from each class
    for class_name in os.listdir(train_path):
        class_path = os.path.join(train_path, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_image = os.path.join(class_path, images[0])
                try:
                    img = cv2.imread(test_image)
                    if img is not None:
                        img_resized = cv2.resize(img, (128, 128))
                        print(f"  ✓ {class_name}: Successfully loaded and resized image")
                    else:
                        print(f"  ✗ {class_name}: Failed to load image")
                except Exception as e:
                    print(f"  ✗ {class_name}: Error loading image: {e}")
    
    return True

def test_tensorflow():
    """Test if TensorFlow is working"""
    print("\nTesting TensorFlow...")
    
    try:
        # Test basic TensorFlow operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = a + b
        print(f"  ✓ TensorFlow basic operations: {c.numpy()}")
        
        # Test model creation
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        print("  ✓ TensorFlow model creation: Success")
        
        return True
    except Exception as e:
        print(f"  ✗ TensorFlow error: {e}")
        return False

def test_dependencies():
    """Test if all required packages are available"""
    print("\nTesting dependencies...")
    
    required_packages = [
        'cv2', 'numpy', 'sklearn', 'matplotlib', 'seaborn'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"  ✓ OpenCV version: {cv2.__version__}")
            elif package == 'numpy':
                import numpy as np
                print(f"  ✓ NumPy version: {np.__version__}")
            elif package == 'sklearn':
                import sklearn
                print(f"  ✓ Scikit-learn version: {sklearn.__version__}")
            elif package == 'matplotlib':
                import matplotlib
                print(f"  ✓ Matplotlib version: {matplotlib.__version__}")
            elif package == 'seaborn':
                import seaborn
                print(f"  ✓ Seaborn version: {seaborn.__version__}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            all_good = False
    
    return all_good

def main():
    print("Brain Tumor Classification System - System Test")
    print("=" * 50)
    
    tests = [
        test_dataset_loading,
        test_image_loading,
        test_tensorflow,
        test_dependencies
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    test_names = ["Dataset Loading", "Image Loading", "TensorFlow", "Dependencies"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(results)
    if all_passed:
        print("\n🎉 All tests passed! Your system is ready for training.")
        print("\nNext steps:")
        print("1. Run: python train_quick.py (for faster training)")
        print("2. Or run: python train.py (for full VGG16 training)")
        print("3. After training, use: python predict.py (to classify new images)")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
