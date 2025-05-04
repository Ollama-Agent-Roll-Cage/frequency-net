import os
import sys
import glob
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Import the StyleGAN3 preprocessor
from stylegan3_pre_processor import StyleGAN3PreProcessor


def process_dataset(input_dir, output_dir, model_path=None, batch_size=4):
    """
    Process all images in a directory with the StyleGAN3 preprocessor.
    This function is specifically designed for CycleGAN datasets.
    
    Args:
        input_dir: Base directory containing trainA, trainB, testA, testB subdirectories
        output_dir: Base directory to save processed images
        model_path: Path to trained model weights (optional)
        batch_size: Batch size for processing
    """
    # Verify input directory structure
    for subdir in ['trainA', 'trainB', 'testA', 'testB']:
        if not os.path.exists(os.path.join(input_dir, subdir)):
            print(f"Warning: {os.path.join(input_dir, subdir)} not found")
    
    # Create output directories
    for subdir in ['trainA', 'trainB', 'testA', 'testB']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = StyleGAN3PreProcessor().to(device)
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    
    # Set to evaluation mode
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Process each subdirectory
    for subdir in ['trainA', 'trainB', 'testA', 'testB']:
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)
        
        # Skip if directory doesn't exist
        if not os.path.exists(input_subdir):
            continue
        
        # Get all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(input_subdir, ext)))
        
        print(f"Processing {len(image_files)} images in {subdir}")
        
        # Process images in batches
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i:i+batch_size]
            batch_tensors = []
            
            # Load and preprocess images
            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    tensor = transform(img)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            if not batch_tensors:
                continue
                
            # Stack into a batch
            batch = torch.stack(batch_tensors).to(device)
            
            # Process batch
            with torch.no_grad():
                outputs = model(batch)
            
            # Save outputs
            for j, output in enumerate(outputs):
                if j >= len(batch_files):
                    break
                    
                # Convert back to image
                output = output.cpu().detach()
                output = ((output * 0.5) + 0.5).clamp(0, 1)
                output_np = output.permute(1, 2, 0).numpy() * 255
                output_img = Image.fromarray(output_np.astype(np.uint8))
                
                # Save to output directory with same filename
                img_name = os.path.basename(batch_files[j])
                output_path = os.path.join(output_subdir, img_name)
                output_img.save(output_path)
    
    print(f"All images processed and saved to {output_dir}")


def before_after_comparison(input_dir, output_dir, comparison_dir, num_samples=5):
    """
    Create before/after comparison images for a processed dataset.
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory containing processed images
        comparison_dir: Directory to save comparison images
        num_samples: Number of sample pairs to create
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Select subdirectories to sample from
    subdirs = []
    for subdir in ['trainA', 'trainB', 'testA', 'testB']:
        if os.path.exists(os.path.join(input_dir, subdir)) and os.path.exists(os.path.join(output_dir, subdir)):
            subdirs.append(subdir)
    
    if not subdirs:
        print("No matching subdirectories found for comparison")
        return
    
    # Process each subdirectory
    for subdir in subdirs:
        # Get all image files
        input_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            input_files.extend(glob.glob(os.path.join(input_dir, subdir, ext)))
        
        if not input_files:
            continue
            
        # Select random samples
        import random
        samples = random.sample(input_files, min(num_samples, len(input_files)))
        
        for i, input_path in enumerate(samples):
            img_name = os.path.basename(input_path)
            output_path = os.path.join(output_dir, subdir, img_name)
            
            if not os.path.exists(output_path):
                print(f"Processed image not found: {output_path}")
                continue
            
            # Load images
            input_img = Image.open(input_path).convert('RGB')
            output_img = Image.open(output_path).convert('RGB')
            
            # Create comparison figure
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(np.array(input_img))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(np.array(output_img))
            plt.title('StyleGAN3-Processed')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save comparison
            comparison_path = os.path.join(comparison_dir, f"{subdir}_{i}_{img_name}")
            plt.savefig(comparison_path, dpi=150)
            plt.close()
            
            print(f"Created comparison: {comparison_path}")
    
    print(f"Comparison images saved to {comparison_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images with StyleGAN3 preprocessor")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, choices=["process", "compare"],
                       help="Mode: 'process' to process a dataset or 'compare' to create comparisons")
    
    # Common arguments
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory to save processed images")
    
    # Processing arguments
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model weights (optional)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing")
    
    # Comparison arguments
    parser.add_argument("--comparison_dir", type=str, default="comparisons",
                       help="Directory to save comparison images")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of sample pairs to create for comparison")
    
    args = parser.parse_args()
    
    if args.mode == "process":
        process_dataset(args.input_dir, args.output_dir, args.model_path, args.batch_size)
    elif args.mode == "compare":
        before_after_comparison(args.input_dir, args.output_dir, args.comparison_dir, args.num_samples)
