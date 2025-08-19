import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter
import random
from scipy import ndimage
import matplotlib.pyplot as plt

class PhotometricAugmentation:
    def __init__(self, seed=None):
        """Initialize the photometric augmentation class."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Augmentation probabilities
        self.prob_gamma = 0.3
        self.prob_contrast = 0.3
        self.prob_clahe = 0.3
        self.prob_noise = 0.3
        self.prob_blur = 0.2
        self.prob_sharpen = 0.2
    
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
            # Apply gamma correction
            image = np.power(image, gamma)
            print(f"Applied gamma jitter: γ = {gamma:.3f}")
        
        return image
    
    def contrast_jitter(self, image, contrast_range=0.1):
        """Apply contrast jitter."""
        if random.random() < self.prob_contrast:
            contrast_factor = random.uniform(1 - contrast_range, 1 + contrast_range)
            # Apply contrast adjustment
            mean_val = np.mean(image)
            image = (image - mean_val) * contrast_factor + mean_val
            print(f"Applied contrast jitter: factor = {contrast_factor:.3f}")
        
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
                # Apply to each channel separately
                img_clahe = np.zeros_like(img_uint8)
                for i in range(img_uint8.shape[2]):
                    img_clahe[:, :, i] = clahe.apply(img_uint8[:, :, i])
            else:
                img_clahe = clahe.apply(img_uint8)
            
            # Convert back to float [0,1]
            image = img_clahe.astype(np.float32) / 255.0
            print(f"Applied CLAHE: clip_limit = {clip_limit:.3f}, tile_size = {tile_size}x{tile_size}")
        
        return image
    
    def add_noise(self, image, noise_range=(0.01, 0.03)):
        """Add Rician/Gaussian noise."""
        if random.random() < self.prob_noise:
            # Calculate dynamic range
            dynamic_range = np.max(image) - np.min(image)
            sigma = random.uniform(*noise_range) * dynamic_range
            
            # Add Gaussian noise
            noise = np.random.normal(0, sigma, image.shape)
            image = image + noise
            print(f"Added noise: σ = {sigma:.6f}")
        
        return image
    
    def gaussian_blur(self, image, sigma_range=(0.3, 0.7)):
        """Apply Gaussian blur."""
        if random.random() < self.prob_blur:
            sigma = random.uniform(*sigma_range)
            # Apply Gaussian blur
            image = ndimage.gaussian_filter(image, sigma=sigma)
            print(f"Applied Gaussian blur: σ = {sigma:.3f} px")
        
        return image
    
    def sharpen(self, image, amount=0.5, radius_range=(1, 2)):
        """Apply unsharp mask sharpening."""
        if random.random() < self.prob_sharpen:
            radius = random.randint(int(radius_range[0]), int(radius_range[1]))
            
            # Use OpenCV for sharpening instead of PIL
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_uint8 = (image * 255).astype(np.uint8)
            img_sharp = cv2.filter2D(img_uint8, -1, kernel)
            
            # Convert back to float [0,1]
            image = img_sharp.astype(np.float32) / 255.0
            print(f"Applied sharpening: radius = {radius} px")
        
        return image
    
    def augment_image(self, image, num_augmentations=2):
        """Apply photometric augmentation to image."""
        print(f"Starting photometric augmentation with {num_augmentations} techniques...")
        
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
        
        print("Augmentation completed!")
        return image
    
    def save_augmented_image(self, image, output_path, original_name):
        """Save augmented image with descriptive filename."""
        # Convert back to uint8 for saving
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Create output filename
        base_name = os.path.splitext(original_name)[0]
        output_filename = f"{base_name}_photometric_aug.jpg"
        full_output_path = os.path.join(output_path, output_filename)
        
        # Save image
        cv2.imwrite(full_output_path, img_uint8)
        print(f"Saved augmented image: {full_output_path}")
        
        return full_output_path

def main():
    """Main function to run the photometric augmentation."""
    # Input and output paths
    input_image_path = "Test/Te-gl_0010.jpg"
    output_dir = "Test"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input image exists
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at {input_image_path}")
        return
    
    # Load image
    print(f"Loading image: {input_image_path}")
    image = cv2.imread(input_image_path)
    
    if image is None:
        print("Error: Could not load image")
        return
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create augmentation object
    aug = PhotometricAugmentation(seed=42)
    
    # Apply augmentation
    augmented_image = aug.augment_image(image, num_augmentations=2)
    
    # Save augmented image
    output_path = aug.save_augmented_image(augmented_image, output_dir, "Te-gl_0010.jpg")
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Augmented image
    plt.subplot(1, 3, 2)
    plt.imshow(augmented_image)
    plt.title("Augmented Image")
    plt.axis('off')
    
    # Difference
    plt.subplot(1, 3, 3)
    diff = np.abs(image.astype(np.float32) - augmented_image)
    plt.imshow(diff, cmap='hot')
    plt.title("Difference (Absolute)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "augmentation_comparison.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nAugmentation completed successfully!")
    print(f"Original image: {input_image_path}")
    print(f"Augmented image: {output_path}")
    print(f"Comparison plot: {os.path.join(output_dir, 'augmentation_comparison.png')}")

if __name__ == "__main__":
    main()
