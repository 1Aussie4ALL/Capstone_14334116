import os
import shutil
import random
import cv2
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import map_coordinates
import skimage.transform as transform

class VariationBDatasetCreator800:
    def __init__(self, seed=42):
        """Initialize the Variation B dataset creator for 800 images per class."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Source and destination paths
        self.dataset_path = "../Dataset/Training"
        self.data_used_path = "data_used_for_variation_A_800"  # Using same source as Variation A
        self.results_path = "Variation B_results_800"
        
        # Cancer types
        self.cancer_types = ['glioma', 'meningioma', 'notumor', 'pituitary', 'not_mri']
        
        # Geometric augmentation probabilities (p≈0.3 each, as specified)
        self.prob_rotate = 0.3
        self.prob_translate = 0.3
        self.prob_scale = 0.3
        self.prob_hflip = 0.3
        self.prob_elastic = 0.2
        
        # Number of images to process per class
        self.images_per_class = 800
        
    def copy_original_images(self):
        """Copy original images to data_used_for_variation_A_800 folder (reusing from Variation A)."""
        print("📁 Using existing data from data_used_for_variation_A_800...")
        
        # Check if source data exists
        if not os.path.exists(self.data_used_path):
            print("⚠️ Source data not found. Please run Variation A first or copy data manually.")
            return False
        
        print("✅ Source data found and ready for Variation B processing!")
        return True
    
    def rotate_image(self, image, angle_range=(-7, 7), cap_range=(-10, 10)):
        """Apply rotation augmentation."""
        if random.random() < self.prob_rotate:
            # Occasionally extend range to cap
            if random.random() < 0.2:  # 20% chance for extended range
                angle = random.uniform(*cap_range)
            else:
                angle = random.uniform(*angle_range)
            
            # Get image center
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                  borderMode=cv2.BORDER_REFLECT_101)
        
        return image
    
    def translate_image(self, image, translate_percent=0.03):
        """Apply translation augmentation."""
        if random.random() < self.prob_translate:
            height, width = image.shape[:2]
            
            # Calculate translation in pixels (≤3% of width/height)
            max_tx = int(width * translate_percent)
            max_ty = int(height * translate_percent)
            
            tx = random.randint(-max_tx, max_tx)
            ty = random.randint(-max_ty, max_ty)
            
            # Create translation matrix
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            
            # Apply translation
            image = cv2.warpAffine(image, translation_matrix, (width, height), 
                                  borderMode=cv2.BORDER_REFLECT_101)
        
        return image
    
    def scale_image(self, image, scale_range=(0.95, 1.05), cap_range=(0.9, 1.1)):
        """Apply scaling/zooming augmentation."""
        if random.random() < self.prob_scale:
            # Occasionally extend range to cap
            if random.random() < 0.2:  # 20% chance for extended range
                scale = random.uniform(*cap_range)
            else:
                scale = random.uniform(*scale_range)
            
            height, width = image.shape[:2]
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # If scaled down, pad to original size
            if scale < 1.0:
                # Create black background
                result = np.zeros_like(image)
                
                # Calculate padding
                pad_x = (width - new_width) // 2
                pad_y = (height - new_height) // 2
                
                # Place scaled image in center
                result[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = scaled_image
                image = result
            else:
                # If scaled up, crop to original size
                start_x = (new_width - width) // 2
                start_y = (new_height - height) // 2
                image = scaled_image[start_y:start_y+height, start_x:start_x+width]
        
        return image
    
    def horizontal_flip(self, image):
        """Apply horizontal flip augmentation."""
        if random.random() < self.prob_hflip:
            # For MRI images, horizontal flip is usually safe for tumor type classification
            # as it doesn't change the anatomical structure significantly
            image = cv2.flip(image, 1)  # 1 for horizontal flip
        
        return image
    
    def elastic_deformation(self, image, alpha_range=(1, 3), sigma_range=(6, 8)):
        """Apply light elastic/affine deformation."""
        if random.random() < self.prob_elastic:
            alpha = random.uniform(*alpha_range)
            sigma = random.uniform(*sigma_range)
            
            height, width = image.shape[:2]
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            
            # Create random displacement fields
            dx = np.random.randn(height, width) * alpha
            dy = np.random.randn(height, width) * alpha
            
            # Apply Gaussian smoothing to displacement fields
            dx = ndimage.gaussian_filter(dx, sigma=sigma)
            dy = ndimage.gaussian_filter(dy, sigma=sigma)
            
            # Normalize displacement fields
            dx = dx * alpha / np.max(np.abs(dx)) if np.max(np.abs(dx)) > 0 else dx
            dy = dy * alpha / np.max(np.abs(dy)) if np.max(np.abs(dy)) > 0 else dy
            
            # Apply displacement
            x_displaced = x + dx
            y_displaced = y + dy
            
            # Map coordinates
            if len(image.shape) == 3:
                # Color image
                result = np.zeros_like(image)
                for channel in range(image.shape[2]):
                    result[:, :, channel] = map_coordinates(image[:, :, channel], 
                                                          [y_displaced, x_displaced], 
                                                          order=1, mode='reflect')
            else:
                # Grayscale image
                result = map_coordinates(image, [y_displaced, x_displaced], 
                                      order=1, mode='reflect')
            
            image = result.astype(image.dtype)
        
        return image
    
    def augment_image(self, image, num_augmentations=2):
        """Apply geometric augmentation to image."""
        # Apply augmentations
        augmentations = [
            self.rotate_image,
            self.translate_image,
            self.scale_image,
            self.horizontal_flip,
            self.elastic_deformation
        ]
        
        # Randomly select 1-2 augmentations (as specified)
        num_augs = random.randint(1, 2)
        selected_augs = random.sample(augmentations, min(num_augs, len(augmentations)))
        
        for aug_func in selected_augs:
            image = aug_func(image)
        
        return image
    
    def create_variation_b_images(self):
        """Create Variation B augmented images for all copied images."""
        print("🎨 Creating Variation B (Geometric) augmented images...")
        
        for cancer_type in self.cancer_types:
            source_dir = os.path.join(self.data_used_path, cancer_type)
            dest_dir = os.path.join(self.results_path, cancer_type)
            
            if not os.path.exists(source_dir):
                print(f"⚠️ Source directory not found: {source_dir}")
                continue
            
            # Create destination directory if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)
            
            # Get list of copied images
            image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"🎨 {cancer_type}: Processing {len(image_files)} images...")
            
            for i, image_file in enumerate(image_files):
                source_path = os.path.join(source_dir, image_file)
                
                # Create output filename
                base_name = os.path.splitext(image_file)[0]
                output_filename = f"{base_name}_variation_B.jpg"
                dest_path = os.path.join(dest_dir, output_filename)
                
                try:
                    # Load image
                    image = cv2.imread(source_path)
                    if image is None:
                        print(f"   ❌ Could not load {image_file}")
                        continue
                    
                    # Apply geometric augmentation
                    augmented_image = self.augment_image(image, num_augmentations=2)
                    
                    # Save augmented image
                    cv2.imwrite(dest_path, augmented_image)
                    
                    if (i + 1) % 100 == 0:
                        print(f"   ✅ Processed {i + 1}/{len(image_files)} images")
                        
                except Exception as e:
                    print(f"   ❌ Error processing {image_file}: {e}")
            
            print(f"✅ Completed Variation B for {cancer_type}: {len(image_files)} images")
        
        print("🎨 Variation B (Geometric) image creation completed!")
    
    def create_summary_report(self):
        """Create a summary report of the process."""
        print("\n📊 Creating summary report...")
        
        report = []
        report.append("=" * 70)
        report.append("VARIATION B DATASET CREATION SUMMARY (800 per class)")
        report.append("GEOMETRIC AUGMENTATION - ANATOMICALLY PLAUSIBLE")
        report.append("=" * 70)
        
        total_original = 0
        total_augmented = 0
        
        for cancer_type in self.cancer_types:
            data_used_dir = os.path.join(self.data_used_path, cancer_type)
            results_dir = os.path.join(self.results_path, cancer_type)
            
            original_count = len([f for f in os.listdir(data_used_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) if os.path.exists(data_used_dir) else 0
            augmented_count = len([f for f in os.listdir(results_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) if os.path.exists(results_dir) else 0
            
            total_original += original_count
            total_augmented += augmented_count
            
            report.append(f"{cancer_type.upper():<15}: {original_count:>4} original -> {augmented_count:>4} augmented")
        
        report.append("-" * 70)
        report.append(f"TOTAL{'':<10}: {total_original:>4} original -> {total_augmented:>4} augmented")
        report.append("=" * 70)
        
        # Add augmentation details
        report.append("\nGEOMETRIC AUGMENTATION DETAILS:")
        report.append("-" * 40)
        report.append("• Rotate: ±7° (cap at ±10°) - p≈0.3")
        report.append("• Translate: ≤3% of width/height - p≈0.3")
        report.append("• Scale/zoom: 0.95–1.05 (cap at 0.9–1.1) - p≈0.3")
        report.append("• Horizontal flip: p=0.3")
        report.append("• Light elastic/affine: α≈1–3 px, σ≈6–8 px - p=0.2")
        report.append("• 1-2 augmentations per image (sampled)")
        report.append("• No vertical flips (anatomically implausible)")
        report.append("• Combined transforms kept subtle for physiological realism")
        
        # Save report
        with open("Variation_B_800_Summary_Report.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(report))
        
        # Print report
        for line in report:
            print(line)
        
        print(f"\n📄 Summary report saved to: Variation_B_800_Summary_Report.txt")
    
    def run_full_process(self):
        """Run the complete Variation B dataset creation process."""
        print("🚀 Starting Variation B Dataset Creation Process (800 per class)...")
        print(f"📁 Target: {self.images_per_class} images per cancer type")
        print(f"🎯 Cancer types: {', '.join(self.cancer_types)}")
        print(f"📊 Expected total: {self.images_per_class * len(self.cancer_types)} images")
        print(f"🔧 Augmentation type: Geometric (anatomically plausible)")
        print()
        
        # Step 1: Check source data
        if not self.copy_original_images():
            print("❌ Cannot proceed without source data. Please run Variation A first.")
            return
        
        print()
        
        # Step 2: Create Variation B augmented images
        self.create_variation_b_images()
        print()
        
        # Step 3: Create summary report
        self.create_summary_report()
        
        print("\n🎉 Variation B Dataset Creation Process Completed!")
        print(f"📁 Source images: {self.data_used_path}")
        print(f"🎨 Augmented images: {self.results_path}")

def main():
    """Main function to run the Variation B dataset creation."""
    creator = VariationBDatasetCreator800(seed=42)
    creator.run_full_process()

if __name__ == "__main__":
    main()
