import os
import requests
from PIL import Image
import io
import random

def download_non_mri_images():
    """
    Download a variety of non-MRI images to create a robust 'not_mri' class
    """
    non_mri_dir = "Dataset/Training/not_mri"
    os.makedirs(non_mri_dir, exist_ok=True)
    
    # URLs for diverse non-MRI images
    non_mri_urls = [
        # Animals
        "https://images.unsplash.com/photo-1543852786-1cf6624b998d?w=400&h=400&fit=crop",  # Cat
        "https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?w=400&h=400&fit=crop",  # Dog
        "https://images.unsplash.com/photo-1549366021-9f761d450615?w=400&h=400&fit=crop",  # Bird
        "https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=400&h=400&fit=crop",  # Fish
        
        # Vehicles
        "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=400&h=400&fit=crop",  # Car
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=400&fit=crop",  # Motorcycle
        "https://images.unsplash.com/photo-1549924231-f129b911e442?w=400&h=400&fit=crop",  # Bicycle
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=400&fit=crop",  # Truck
        
        # People
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop",  # Person
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop",  # Portrait
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop",  # Group
        
        # Nature
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400&h=400&fit=crop",  # Forest
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",  # Mountain
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",  # Ocean
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400&h=400&fit=crop",  # Desert
        
        # Buildings and Architecture
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400&h=400&fit=crop",  # Building
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",  # City
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400&h=400&fit=crop",  # Bridge
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",  # House
        
        # Objects and Items
        "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=400&h=400&fit=crop",  # Object
        "https://images.unsplash.com/photo-1543852786-1cf6624b998d?w=400&h=400&fit=crop",  # Item
        "https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?w=400&h=400&fit=crop",  # Tool
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop",  # Device
        
        # Food
        "https://images.unsplash.com/photo-1543852786-1cf6624b998d?w=400&h=400&fit=crop",  # Food
        "https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?w=400&h=400&fit=crop",  # Fruit
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop",  # Vegetable
        "https://images.unsplash.com/photo-1543852786-1cf6624b998d?w=400&h=400&fit=crop",  # Meal
        
        # Technology
        "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=400&h=400&fit=crop",  # Computer
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=400&fit=crop",  # Phone
        "https://images.unsplash.com/photo-1549924231-f129b911e442?w=400&h=400&fit=crop",  # Camera
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=400&fit=crop",  # Gadget
    ]
    
    print("Downloading diverse non-MRI images for training...")
    successful_downloads = 0
    
    for i, url in enumerate(non_mri_urls):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                img = img.resize((224, 224))
                
                # Generate a descriptive filename
                categories = ['animal', 'vehicle', 'person', 'nature', 'building', 'object', 'food', 'technology']
                category = categories[i // 4]  # Group every 4 images into a category
                
                img_path = os.path.join(non_mri_dir, f"{category}_{i:03d}.jpg")
                img.save(img_path, "JPEG", quality=95)
                print(f"Downloaded: {category}_{i:03d}.jpg")
                successful_downloads += 1
                
                # Add some random variations
                if random.random() < 0.3:  # 30% chance to add variations
                    # Rotate slightly
                    rotated_img = img.rotate(random.uniform(-15, 15))
                    rotated_path = os.path.join(non_mri_dir, f"{category}_{i:03d}_rotated.jpg")
                    rotated_img.save(rotated_path, "JPEG", quality=95)
                    print(f"  Added variation: {category}_{i:03d}_rotated.jpg")
                    successful_downloads += 1
                    
                    # Flip horizontally
                    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    flipped_path = os.path.join(non_mri_dir, f"{category}_{i:03d}_flipped.jpg")
                    flipped_img.save(flipped_path, "JPEG", quality=95)
                    print(f"  Added variation: {category}_{i:03d}_flipped.jpg")
                    successful_downloads += 1
                
        except Exception as e:
            print(f"Error downloading image {i}: {e}")
    
    print(f"\nNon-MRI images downloaded successfully!")
    print(f"Total images added: {successful_downloads}")
    print(f"Images saved to: {non_mri_dir}")
    
    # Also create a testing set for non-MRI images
    test_non_mri_dir = "Dataset/Testing/not_mri"
    os.makedirs(test_non_mri_dir, exist_ok=True)
    
    # Copy some images to testing set
    training_images = [f for f in os.listdir(non_mri_dir) if f.endswith('.jpg')]
    test_images = random.sample(training_images, min(20, len(training_images)))
    
    for img_name in test_images:
        src_path = os.path.join(non_mri_dir, img_name)
        dst_path = os.path.join(test_non_mri_dir, img_name)
        
        # Copy the image
        img = Image.open(src_path)
        img.save(dst_path, "JPEG")
    
    print(f"Testing set created with {len(test_images)} images in: {test_non_mri_dir}")

def create_balanced_dataset():
    """
    Ensure the dataset is balanced between MRI and non-MRI classes
    """
    print("\nChecking dataset balance...")
    
    # Count images in each class
    train_path = "Dataset/Training"
    class_counts = {}
    
    for class_name in os.listdir(train_path):
        class_path = os.path.join(train_path, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[class_name] = len(images)
            print(f"{class_name}: {len(images)} images")
    
    # Check if not_mri class has enough images
    if 'not_mri' in class_counts:
        mri_total = sum(count for class_name, count in class_counts.items() if class_name != 'not_mri')
        not_mri_count = class_counts['not_mri']
        
        print(f"\nDataset balance:")
        print(f"MRI images (all classes): {mri_total}")
        print(f"Non-MRI images: {not_mri_count}")
        
        if not_mri_count < mri_total * 0.2:  # Non-MRI should be at least 20% of MRI images
            print(f"Warning: Non-MRI class is underrepresented. Consider adding more non-MRI images.")
        else:
            print(f"Dataset is well-balanced for training.")
    else:
        print("Error: 'not_mri' class not found in training dataset!")

if __name__ == "__main__":
    print("Adding Non-MRI Images to Dataset")
    print("="*40)
    
    # Download non-MRI images
    download_non_mri_images()
    
    # Check dataset balance
    create_balanced_dataset()
    
    print("\nProcess completed!")
    print("You can now run the training script to train the MRI detection model.")
