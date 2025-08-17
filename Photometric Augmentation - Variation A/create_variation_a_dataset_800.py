import os
import shutil
import random
import cv2
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from scipy import ndimage

class VariationADatasetCreator800:
    def __init__(self, seed=42):
        """Initialize the Variation A dataset creator for 800 images per class."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Source and destination paths
        self.dataset_path = "../Dataset/Training"
        self.data_used_path = "data_used_for_variation_A_800"
        self.results_path = "Variation A_results_800"
        
        # Cancer types
        self.cancer_types = ['glioma', 'meningioma', 'notumor', 'pituitary', 'not_mri']
        
        # Augmentation probabilities (p≈0.3 each)
        self.prob_gamma = 0.3
        self.prob_contrast = 0.3
        self.prob_clahe = 0.3
        self.prob_noise = 0.3
        self.prob_blur = 0.2
        self.prob_sharpen = 0.2
        
        # Number of images to process per class
        self.images_per_class = 800
        
    def copy_original_images(self):
        """Copy original images to data_used_for_variation_A_800 folder."""
        print("📁 Copying original images to data_used_for_variation_A_800...")
        
        for cancer_type in self.cancer_types:
            source_dir = os.path.join(self.dataset_path, cancer_type)
            dest_dir = os.path.join(self.data_used_path, cancer_type)
            
            if not os.path.exists(source_dir):
                print(f"⚠️ Source directory not found: {source_dir}")
                continue
                
            # Get list of all images in source directory
            image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if len(image_files) < self.images_per_class:
                print(f"⚠️ Only {len(image_files)} images found in {cancer_type}, using all available")
                selected_images = image_files
            else:
                # Randomly select images_per_class images
                selected_images = random.sample(image_files, self.images_per_class)
            
            print(f"📋 {cancer_type}: Copying {len(selected_images)} images...")
            
            for i, image_file in enumerate(selected_images):
                source_path = os.path.join(source_dir, image_file)
                dest_path = os.path.join(dest_dir, f"{cancer_type}_{i+1:04d}_{image_file}")
                
                try:
                    shutil.copy2(source_path, dest_path)
                    if (i + 1) % 100 == 0:
                        print(f"   ✅ Copied {i + 1}/{len(selected_images)} images")
                except Exception as e:
                    print(f"   ❌ Error copying {image_file}: {e}")
            
            print(f"✅ Completed copying {cancer_type}: {len(selected_images)} images")
        
        print("📁 Original image copying completed!")
    
    def normalize_image(self, image):
        """Normalize image to [0,1] range."""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        return image
    
    def clip_to_range(self, image):
        """Clip image values to [0,1] range."""
        return np.clip(image, 0.0, 1.0)
    
    def gamma_jitter(self, image, gamma_range=(0.9, 1.1)):
        """Apply gamma (brightness) jitter."""
        if random.random() < self.prob_gamma:
            # Occasionally extend range to 0.8-1.2
            if random.random() < 0.2:  # 20% chance for extended range
                gamma_range = (0.8, 1.2)
            
            gamma = random.uniform(*gamma_range)
            image = np.power(image, gamma)
        
        return image
    
    def contrast_jitter(self, image, contrast_range=0.1):
        """Apply contrast jitter."""
        if random.random() < self.prob_contrast:
            contrast_factor = random.uniform(1 - contrast_range, 1 + contrast_range)
            mean_val = np.mean(image)
            image = (image - mean_val) * contrast_factor + mean_val
        
        return image
    
    def apply_clahe(self, image, clip_limit_range=(1.5, 2.0), tile_size=8):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        if random.random() < self.prob_clahe:
            clip_limit = random.uniform(*clip_limit_range)
            
            # Convert to uint8 for CLAHE
            img_uint8 = (image * 255).astype(np.uint8)
            
            # Create CLAHE object
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            
            # Apply CLAHE
            if len(img_uint8.shape) == 3:
                img_clahe = np.zeros_like(img_uint8)
                for i in range(img_uint8.shape[2]):
                    img_clahe[:, :, i] = clahe.apply(img_uint8[:, :, i])
            else:
                img_clahe = clahe.apply(img_uint8)
            
            # Convert back to float [0,1]
            image = img_clahe.astype(np.float32) / 255.0
        
        return image
    
    def add_noise(self, image, noise_range=(0.01, 0.03)):
        """Add Rician/Gaussian noise."""
        if random.random() < self.prob_noise:
            dynamic_range = np.max(image) - np.min(image)
            sigma = random.uniform(*noise_range) * dynamic_range
            noise = np.random.normal(0, sigma, image.shape)
            image = image + noise
        
        return image
    
    def gaussian_blur(self, image, sigma_range=(0.3, 0.7)):
        """Apply Gaussian blur."""
        if random.random() < self.prob_blur:
            sigma = random.uniform(*sigma_range)
            image = ndimage.gaussian_filter(image, sigma=sigma)
        
        return image
    
    def sharpen(self, image, amount=0.5, radius_range=(1, 2)):
        """Apply unsharp mask sharpening."""
        if random.random() < self.prob_sharpen:
            radius = random.randint(int(radius_range[0]), int(radius_range[1]))
            
            # Use OpenCV for sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_uint8 = (image * 255).astype(np.uint8)
            img_sharp = cv2.filter2D(img_uint8, -1, kernel)
            
            # Convert back to float [0,1]
            image = img_sharp.astype(np.float32) / 255.0
        
        return image
    
    def augment_image(self, image, num_augmentations=2):
        """Apply photometric augmentation to image."""
        # Normalize image first
        image = self.normalize_image(image)
        
        # Apply augmentations
        augmentations = [
            self.gamma_jitter,
            self.contrast_jitter,
            self.apply_clahe,
            self.add_noise,
            self.gaussian_blur,
            self.sharpen
        ]
        
        # Randomly select augmentations
        selected_augs = random.sample(augmentations, min(num_augmentations, len(augmentations)))
        
        for aug_func in selected_augs:
            image = aug_func(image)
        
        # Re-clip to [0,1] range
        image = self.clip_to_range(image)
        
        return image
    
    def create_variation_a_images(self):
        """Create Variation A augmented images for all copied images."""
        print("🎨 Creating Variation A augmented images...")
        
        for cancer_type in self.cancer_types:
            source_dir = os.path.join(self.data_used_path, cancer_type)
            dest_dir = os.path.join(self.results_path, cancer_type)
            
            if not os.path.exists(source_dir):
                print(f"⚠️ Source directory not found: {source_dir}")
                continue
            
            # Get list of copied images
            image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"🎨 {cancer_type}: Processing {len(image_files)} images...")
            
            for i, image_file in enumerate(image_files):
                source_path = os.path.join(source_dir, image_file)
                
                # Create output filename
                base_name = os.path.splitext(image_file)[0]
                output_filename = f"{base_name}_variation_A.jpg"
                dest_path = os.path.join(dest_dir, output_filename)
                
                try:
                    # Load image
                    image = cv2.imread(source_path)
                    if image is None:
                        print(f"   ❌ Could not load {image_file}")
                        continue
                    
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Apply augmentation
                    augmented_image = self.augment_image(image, num_augmentations=2)
                    
                    # Save augmented image
                    img_uint8 = (augmented_image * 255).astype(np.uint8)
                    cv2.imwrite(dest_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
                    
                    if (i + 1) % 100 == 0:
                        print(f"   ✅ Processed {i + 1}/{len(image_files)} images")
                        
                except Exception as e:
                    print(f"   ❌ Error processing {image_file}: {e}")
            
            print(f"✅ Completed Variation A for {cancer_type}: {len(image_files)} images")
        
        print("🎨 Variation A image creation completed!")
    
    def create_summary_report(self):
        """Create a summary report of the process."""
        print("\n📊 Creating summary report...")
        
        report = []
        report.append("=" * 70)
        report.append("VARIATION A DATASET CREATION SUMMARY (800 per class)")
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
        
        # Save report
        with open("Variation_A_800_Summary_Report.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(report))
        
        # Print report
        for line in report:
            print(line)
        
        print(f"\n📄 Summary report saved to: Variation_A_800_Summary_Report.txt")
    
    def run_full_process(self):
        """Run the complete Variation A dataset creation process."""
        print("🚀 Starting Variation A Dataset Creation Process (800 per class)...")
        print(f"📁 Target: {self.images_per_class} images per cancer type")
        print(f"🎯 Cancer types: {', '.join(self.cancer_types)}")
        print(f"📊 Expected total: {self.images_per_class * len(self.cancer_types)} images")
        print()
        
        # Step 1: Copy original images
        self.copy_original_images()
        print()
        
        # Step 2: Create Variation A augmented images
        self.create_variation_a_images()
        print()
        
        # Step 3: Create summary report
        self.create_summary_report()
        
        print("\n🎉 Variation A Dataset Creation Process Completed!")
        print(f"📁 Original images: {self.data_used_path}")
        print(f"🎨 Augmented images: {self.results_path}")

def main():
    """Main function to run the Variation A dataset creation."""
    creator = VariationADatasetCreator800(seed=42)
    creator.run_full_process()

if __name__ == "__main__":
    main()
