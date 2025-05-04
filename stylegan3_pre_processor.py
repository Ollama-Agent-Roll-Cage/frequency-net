import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import glob
import numpy as np
from PIL import Image
import argparse
import math


class LowpassFilter(nn.Module):
    """
    Applies a low-pass filter to prevent aliasing, inspired by StyleGAN3.
    """
    def __init__(self, cutoff=0.5, filter_size=6):
        super().__init__()
        self.cutoff = cutoff
        self.filter_size = filter_size
        self.register_buffer('kernel', self._build_filter_kernel())
        
    def _build_filter_kernel(self):
        # Create a 2D sinc filter kernel (low-pass)
        kernel_size = 2 * self.filter_size + 1
        kernel = torch.zeros(kernel_size, kernel_size)
        
        for x in range(kernel_size):
            for y in range(kernel_size):
                dx = (x - self.filter_size) / self.filter_size
                dy = (y - self.filter_size) / self.filter_size
                distance = math.sqrt(dx**2 + dy**2)
                if distance < 1e-6:
                    kernel[x, y] = self.cutoff**2
                else:
                    kernel[x, y] = self.cutoff**2 * (2 * torch.tensor(math.pi * distance).sin() / (math.pi * distance))
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)
    
    def forward(self, x):
        # Apply filtering to each channel independently
        b, c, h, w = x.shape
        x_filtered = F.conv2d(
            x.view(b * c, 1, h, w),
            self.kernel,
            padding=self.filter_size
        ).view(b, c, h, w)
        return x_filtered


class FilteredNonlinearity(nn.Module):
    """
    Applies a non-linearity followed by low-pass filtering to prevent aliasing.
    """
    def __init__(self, cutoff=0.5, filter_size=6):
        super().__init__()
        self.filter = LowpassFilter(cutoff, filter_size)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        return self.filter(self.act(x))


class FourierFeatures(nn.Module):
    """
    Generates Fourier features that preserve equivariance properties.
    """
    def __init__(self, out_channels, input_scale=10.0):
        super().__init__()
        self.out_channels = out_channels
        self.input_scale = input_scale
        # Create frequency multipliers
        self.register_buffer('freq_multiplier', torch.randn(out_channels // 4, 2) * input_scale)
        
    def forward(self, coords):
        """
        Args:
            coords: Coordinate grid of shape [batch, 2, height, width]
        """
        # Scale coordinates
        x = coords * self.freq_multiplier.view(self.out_channels // 4, 2, 1, 1)
        
        # Compute sin and cos features
        sin_features = torch.sin(x)
        cos_features = torch.cos(x)
        
        # Concatenate to get the final features
        features = torch.cat([sin_features, cos_features], dim=1)
        return features.reshape(-1, self.out_channels, coords.shape[2], coords.shape[3])


class StyleGAN3PreProcessor(nn.Module):
    def __init__(self, input_channels=3, fourier_scale=10.0):
        super().__init__()
        self.fourier_scale = fourier_scale
        
        # Process both spatial and color information
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, input_channels, 3, padding=1)
        )
    
    def forward(self, x):
        try:
            # Initial processing
            x = x.float()
            spatial_out = self.main(x)
            
            # FFT processing
            b, c, h, w = spatial_out.shape
            freq_outputs = []
            
            for ch in range(c):
                # Process each channel in frequency domain
                freq = torch.fft.rfft2(spatial_out[:, ch:ch+1], dim=(-2, -1), norm='ortho')
                
                # Create frequency grid
                y_freq = torch.linspace(-1, 1, h, device=x.device)
                x_freq = torch.linspace(-1, 1, w//2 + 1, device=x.device)
                grid_y, grid_x = torch.meshgrid(y_freq, x_freq, indexing='ij')
                dist = torch.sqrt(grid_y**2 + grid_x**2)
                
                # Create mask with different processing for luminance/chrominance
                if ch == 0:  # Luminance
                    mask = torch.exp(-dist * self.fourier_scale) + 0.3
                else:  # Chrominance
                    mask = torch.exp(-dist * (self.fourier_scale * 0.8)) + 0.4
                
                mask = mask.clamp(0, 1).unsqueeze(0).unsqueeze(0)
                
                # Apply frequency processing
                freq_filtered = freq * mask
                freq_out = torch.fft.irfft2(freq_filtered, s=(h, w), dim=(-2, -1), norm='ortho')
                freq_outputs.append(freq_out)
            
            # Combine channels
            freq_out = torch.cat(freq_outputs, dim=1)
            
            # Blend spatial and frequency information
            output = 0.7 * spatial_out + 0.3 * freq_out
            return output
            
        except Exception as e:
            print(f"Error in StyleGAN3PreProcessor forward: {str(e)}")
            return x


def create_low_pass_mask(shape, scale):
    """Creates a radial mask for frequency separation"""
    h, w = shape[-2:]
    coords = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w // 2 + 1)
    ))
    dist = torch.sqrt((coords ** 2).sum(0))
    return torch.exp(-dist * scale)


def preprocess_images(input_dir, output_dir, model_path=None, batch_size=4):
    """
    Process all images in a directory using the StyleGAN3 preprocessor.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        model_path: Path to pretrained model weights (if available)
        batch_size: Batch size for processing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = StyleGAN3PreProcessor().to(device)
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    
    # Set to evaluation mode
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_tensors = []
        
        # Load and preprocess images
        for img_path in batch_files:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)
            batch_tensors.append(tensor)
        
        # Stack into a batch
        batch = torch.stack(batch_tensors).to(device)
        
        # Process batch
        with torch.no_grad():
            outputs = model(batch)
        
        # Save outputs
        for j, output in enumerate(outputs):
            # Convert back to image
            output = output.cpu().detach()
            output = ((output * 0.5) + 0.5).clamp(0, 1)
            output_np = output.permute(1, 2, 0).numpy() * 255
            output_img = Image.fromarray(output_np.astype(np.uint8))
            
            # Save to output directory
            img_name = os.path.basename(batch_files[j])
            output_path = os.path.join(output_dir, img_name)
            output_img.save(output_path)
            
        print(f"Processed {min(i+batch_size, len(image_files))}/{len(image_files)} images")


def train_processor(train_dir, val_dir, output_model_path, epochs=10, batch_size=4, lr=0.0002):
    """
    Train the StyleGAN3 preprocessor on a dataset to improve its performance.
    This is a simple training setup using reconstruction loss to preserve image content
    while enhancing equivariance properties.
    
    Args:
        train_dir: Directory containing training images
        val_dir: Directory containing validation images
        output_model_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = StyleGAN3PreProcessor().to(device)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create datasets
    train_dataset = torch.utils.data.dataset.Dataset(
        root=train_dir,
        transform=transform
    )
    
    val_dataset = torch.utils.data.dataset.Dataset(
        root=val_dir,
        transform=transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.L1Loss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, imgs in enumerate(train_loader):
            imgs = imgs.to(device)
            
            # Forward pass
            outputs = model(imgs)
            
            # Compute loss
            loss = criterion(outputs, imgs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, imgs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_model_path)
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN3-inspired image pre-processor for CycleGAN")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, choices=["preprocess", "train"],
                        help="Mode: 'preprocess' to process images or 'train' to train the model")
    
    # Preprocessing arguments
    parser.add_argument("--input_dir", type=str, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, help="Directory to save processed images")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model weights")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    
    # Training arguments
    parser.add_argument("--train_dir", type=str, help="Directory containing training images")
    parser.add_argument("--val_dir", type=str, help="Directory containing validation images")
    parser.add_argument("--output_model_path", type=str, help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    
    args = parser.parse_args()
    
    if args.mode == "preprocess":
        if not args.input_dir or not args.output_dir:
            parser.error("--input_dir and --output_dir are required for preprocess mode")
        
        preprocess_images(args.input_dir, args.output_dir, args.model_path, args.batch_size)
    
    elif args.mode == "train":
        if not args.train_dir or not args.val_dir or not args.output_model_path:
            parser.error("--train_dir, --val_dir, and --output_model_path are required for train mode")
        
        train_processor(args.train_dir, args.val_dir, args.output_model_path, args.epochs, args.batch_size, args.lr)
