import os
import glob
import shutil
import argparse
from stylegan3_pre_processor import preprocess_images

def prepare_dataset_for_cyclegan(input_dir, cyclegan_dataset_name, test_split=0.1):
    """
    Prepare a dataset for CycleGAN training by processing images through StyleGAN3 preprocessor.
    
    Args:
        input_dir: Directory containing original image pairs (A and B)
        cyclegan_dataset_name: Name for the CycleGAN dataset
        test_split: Fraction of images to use for testing
    """
    # Check if original dataset has the expected structure
    if not os.path.exists(os.path.join(input_dir, 'trainA')) or not os.path.exists(os.path.join(input_dir, 'trainB')):
        raise ValueError("Input directory must contain 'trainA' and 'trainB' subdirectories")
    
    # Create base output directory
    base_output_dir = f"datasets/{cyclegan_dataset_name}"
    
    # Create required directories
    dirs_to_create = [
        f"{base_output_dir}/trainA", 
        f"{base_output_dir}/trainB", 
        f"{base_output_dir}/testA", 
        f"{base_output_dir}/testB",
        "temp/A_processed",
        "temp/B_processed"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process domain A images with StyleGAN3 preprocessor
    print("Processing domain A images...")
    preprocess_images(
        input_dir=os.path.join(input_dir, 'trainA'),
        output_dir="temp/A_processed",
        batch_size=4
    )
    
    # Process domain B images with StyleGAN3 preprocessor
    print("Processing domain B images...")
    preprocess_images(
        input_dir=os.path.join(input_dir, 'trainB'),
        output_dir="temp/B_processed",
        batch_size=4
    )
    
    # Split processed images into train and test sets
    def split_and_copy(src_dir, train_dir, test_dir, test_split=0.1):
        all_files = glob.glob(os.path.join(src_dir, "*.jpg")) + glob.glob(os.path.join(src_dir, "*.png"))
        test_count = max(1, int(len(all_files) * test_split))
        train_count = len(all_files) - test_count
        
        # Copy files to train directory
        for i, file_path in enumerate(all_files[:train_count]):
            dest_path = os.path.join(train_dir, f"{i:05d}.png")
            shutil.copy(file_path, dest_path)
        
        # Copy files to test directory
        for i, file_path in enumerate(all_files[train_count:]):
            dest_path = os.path.join(test_dir, f"{i:05d}.png")
            shutil.copy(file_path, dest_path)
        
        return train_count, test_count
    
    # Split and copy domain A
    a_train, a_test = split_and_copy(
        "temp/A_processed",
        f"{base_output_dir}/trainA",
        f"{base_output_dir}/testA",
        test_split
    )
    
    # Split and copy domain B
    b_train, b_test = split_and_copy(
        "temp/B_processed",
        f"{base_output_dir}/trainB",
        f"{base_output_dir}/testB",
        test_split
    )
    
    print(f"Domain A: {a_train} training images, {a_test} test images")
    print(f"Domain B: {b_train} training images, {b_test} test images")
    
    # Cleanup temporary directories
    shutil.rmtree("temp")
    
    print(f"Dataset prepared at {base_output_dir}")
    print("You can now train CycleGAN using this command:")
    print(f"python train.py --dataroot ./datasets/{cyclegan_dataset_name} --name {cyclegan_dataset_name} --model cycle_gan")


def process_and_train(input_dir, cyclegan_dataset_name, cyclegan_path, epochs=200, batch_size=1):
    """
    Process images with StyleGAN3 preprocessor and train a CycleGAN model.
    
    Args:
        input_dir: Directory containing original image pairs (A and B)
        cyclegan_dataset_name: Name for the CycleGAN dataset
        cyclegan_path: Path to the CycleGAN repository
        epochs: Number of epochs to train for
        batch_size: Batch size for training
    """
    # First prepare the dataset
    prepare_dataset_for_cyclegan(input_dir, cyclegan_dataset_name)
    
    # Change to CycleGAN directory
    original_dir = os.getcwd()
    os.chdir(cyclegan_path)
    
    # Run training command
    train_command = (
        f"python train.py "
        f"--dataroot ./datasets/{cyclegan_dataset_name} "
        f"--name {cyclegan_dataset_name} "
        f"--model cycle_gan "
        f"--n_epochs {epochs//2} "
        f"--n_epochs_decay {epochs//2} "
        f"--batch_size {batch_size}"
    )
    
    print(f"Running training command: {train_command}")
    os.system(train_command)
    
    # Return to original directory
    os.chdir(original_dir)
    
    print(f"Training complete. Model saved at {cyclegan_path}/checkpoints/{cyclegan_dataset_name}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and train CycleGAN with StyleGAN3 preprocessing")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, choices=["prepare", "train"],
                        help="Mode: 'prepare' to process images or 'train' to process and train")
    
    # Common arguments
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing original image pairs (A and B)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name for the CycleGAN dataset")
    
    # Training arguments
    parser.add_argument("--cyclegan_path", type=str, default="./pytorch-CycleGAN-and-pix2pix",
                        help="Path to the CycleGAN repository")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    
    args = parser.parse_args()
    
    if args.mode == "prepare":
        prepare_dataset_for_cyclegan(args.input_dir, args.dataset_name)
    
    elif args.mode == "train":
        process_and_train(args.input_dir, args.dataset_name, args.cyclegan_path, args.epochs, args.batch_size)
