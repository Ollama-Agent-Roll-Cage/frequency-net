from contextlib import nullcontext
import argparse
import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import tempfile
import time
import cv2  # Add this at the top with other imports

# Memory management settings
torch.backends.cudnn.benchmark = True  # Enable for fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

# Set stricter memory limits
MAX_BATCH_SIZE = 1
MIN_IMAGE_SIZE = 32
MAX_IMAGE_SIZE = 2048  # Allow larger images
DEFAULT_IMAGE_SIZE = 512  # Higher default
MEMORY_FRACTION = 0.6  # More memory for processing

# Optional: Set environment variable for blocking CUDA allocation
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Import the StyleGAN3 preprocessor
from stylegan3_pre_processor import StyleGAN3PreProcessor


class StyleGAN3Demo:
    def __init__(self, model_path=None):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
        else:
            self.device = torch.device('cpu')
            
        self.model = StyleGAN3PreProcessor()
        self.model = self.model.to(self.device)
        
        # Initialize transform with default size
        self._update_transform(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        
        print(f"StyleGAN3 demo initialized on {self.device}")
    
    def _update_transform(self, height, width):
        """Update the image transform with correct dimension ordering.
        Args:
            height (int): Target height
            width (int): Target width
        Note: torchvision.transforms.Resize expects (height, width) order
        """
        self.transform = transforms.Compose([
            transforms.Resize((height, width), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def rgb_to_hsv(self, rgb):
        # Convert RGB to HSV
        # Input: RGB image (3, H, W) with values in [0, 1]
        # Output: HSV image (3, H, W) with values in [0, 1]
        
        r, g, b = rgb[0], rgb[1], rgb[2]
        max_rgb, argmax_rgb = torch.max(rgb, dim=0)
        min_rgb, _ = torch.min(rgb, dim=0)
        
        diff = max_rgb - min_rgb
        
        # Hue
        h = torch.zeros_like(max_rgb)
        
        # max_rgb == r
        mask = argmax_rgb == 0
        h[mask] = (60 * (g[mask] - b[mask]) / (diff[mask] + 1e-8) + 360) % 360
        
        # max_rgb == g
        mask = argmax_rgb == 1
        h[mask] = (60 * (b[mask] - r[mask]) / (diff[mask] + 1e-8) + 120)
        
        # max_rgb == b
        mask = argmax_rgb == 2
        h[mask] = (60 * (r[mask] - g[mask]) / (diff[mask] + 1e-8) + 240)
        
        # Saturation
        s = torch.zeros_like(max_rgb)
        mask = max_rgb != 0
        s[mask] = diff[mask] / (max_rgb[mask] + 1e-8)
        
        # Value
        v = max_rgb
        
        return torch.stack([h/360, s, v])

    def hsv_to_rgb(self, hsv):
        # Convert HSV to RGB
        # Input: HSV image (3, H, W) with values in [0, 1]
        # Output: RGB image (3, H, W) with values in [0, 1]
        
        h, s, v = hsv[0] * 360, hsv[1], hsv[2]
        
        c = v * s
        x = c * (1 - torch.abs((h / 60) % 2 - 1))
        m = v - c
        
        rgb = torch.zeros_like(hsv)
        
        # h in [0, 60)
        mask = (h >= 0) & (h < 60)
        rgb[0][mask] = c[mask]
        rgb[1][mask] = x[mask]
        
        # h in [60, 120)
        mask = (h >= 60) & (h < 120)
        rgb[0][mask] = x[mask]
        rgb[1][mask] = c[mask]
        
        # h in [120, 180)
        mask = (h >= 120) & (h < 180)
        rgb[1][mask] = c[mask]
        rgb[2][mask] = x[mask]
        
        # h in [180, 240)
        mask = (h >= 180) & (h < 240)
        rgb[1][mask] = x[mask]
        rgb[2][mask] = c[mask]
        
        # h in [240, 300)
        mask = (h >= 240) & (h < 300)
        rgb[0][mask] = x[mask]
        rgb[2][mask] = c[mask]
        
        # h in [300, 360)
        mask = (h >= 300) & (h < 360)
        rgb[0][mask] = c[mask]
        rgb[2][mask] = x[mask]
        
        rgb += m
        
        return rgb

    def adjust_colors(self, image, hue_shift=0.0, contrast=1.2, saturation=1.0, 
                     brightness=0.0, shadows=0.0, highlights=0.0):
        """Advanced color adjustment with better contrast and brightness control"""
        try:
            # Convert to float
            img = torch.from_numpy(image.astype(np.float32) / 255.0)
            
            # Apply contrast first (more aggressive)
            if contrast != 1.0:
                mean = torch.mean(img, dim=(0, 1), keepdim=True)
                img = torch.sign(img - mean) * (torch.abs(img - mean) ** (1/contrast)) + mean
            
            # Apply brightness
            if brightness != 0:
                img = img + brightness
            
            # Adjust shadows and highlights more aggressively
            if shadows != 0 or highlights != 0:
                luminance = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
                shadow_mask = torch.pow(1.0 - luminance, 2)  # More aggressive shadow adjustment
                highlight_mask = torch.pow(luminance, 2)     # More aggressive highlight adjustment
                
                img = img + (shadows * 1.5) * shadow_mask.unsqueeze(-1)     # Increased effect
                img = img + (highlights * 1.5) * highlight_mask.unsqueeze(-1) # Increased effect
            
            # Convert to HSV for hue and saturation
            hsv = self.rgb_to_hsv(img.permute(2, 0, 1))
            
            # Adjust hue
            hsv[0] = (hsv[0] + hue_shift) % 1.0
            
            # More aggressive saturation
            if saturation != 1.0:
                hsv[1] = torch.clamp(hsv[1] * (saturation * 1.2), 0, 1)  # Increased effect
            
            # Convert back to RGB
            rgb = self.hsv_to_rgb(hsv)
            rgb = rgb.permute(1, 2, 0)
            
            # Final clipping
            rgb = torch.clamp(rgb * 255, 0, 255)
            return rgb.numpy().astype(np.uint8)
            
        except Exception as e:
            print(f"Error adjusting colors: {str(e)}")
            return image
    
    def process_image(self, input_image, target_size, filter_strength=0.75, 
                     fourier_scale=10.0, hue_shift=0.0, contrast=1.0, saturation=1.0, 
                     brightness=0.0, shadows=0.0, highlights=0.0):
        try:
            torch.cuda.empty_cache()
            
            if isinstance(input_image, np.ndarray):
                input_image = Image.fromarray(input_image)
            
            # Calculate dimensions
            orig_width, orig_height = input_image.size
            aspect_ratio = orig_width / orig_height
            
            # Calculate new dimensions
            if orig_width > orig_height:
                new_width = min(target_size, MAX_IMAGE_SIZE)
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = min(target_size, MAX_IMAGE_SIZE)
                new_width = int(new_height * aspect_ratio)
            
            new_width = max(new_width, MIN_IMAGE_SIZE)
            new_height = max(new_height, MIN_IMAGE_SIZE)
            
            print(f"Input size: ({orig_width}, {orig_height})")
            print(f"Output size: ({new_width}, {new_height})")
            
            # Process image
            self._update_transform(new_height, new_width)
            img_tensor = self.transform(input_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                try:
                    output = self.model(img_tensor)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("GPU memory exceeded, falling back to CPU...")
                        torch.cuda.empty_cache()
                        self.model = self.model.cpu()
                        output = self.model(img_tensor.cpu())
                        self.model = self.model.to(self.device)
                    else:
                        raise e
            
            # Convert to numpy
            output = output.cpu()
            output = ((output * 0.5) + 0.5).clamp(0, 1)
            output_np = output.squeeze(0).permute(1, 2, 0).numpy() * 255
            
            # Apply color adjustments
            output_np = self.adjust_colors(
                output_np,
                hue_shift=hue_shift,
                contrast=contrast,
                saturation=saturation,
                brightness=brightness,
                shadows=shadows,
                highlights=highlights
            )
            
            return output_np.astype(np.uint8)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def save_processed_pair(self, input_image, processed_image, output_folder="output"):
        """Save the input and processed images as a pair for comparison"""
        os.makedirs(output_folder, exist_ok=True)
        
        # Create unique filename
        timestamp = int(time.time())
        
        # Convert numpy arrays to PIL images
        if isinstance(input_image, np.ndarray):
            input_pil = Image.fromarray(input_image)
        else:
            input_pil = input_image
            
        if isinstance(processed_image, np.ndarray):
            processed_pil = Image.fromarray(processed_image)
        else:
            processed_pil = processed_image
        
        # Save images
        input_path = os.path.join(output_folder, f"input_{timestamp}.png")
        processed_path = os.path.join(output_folder, f"processed_{timestamp}.png")
        
        input_pil.save(input_path)
        processed_pil.save(processed_path)
        
        print(f"Saved image pair to {input_path} and {processed_path}")
        
        return input_path, processed_path


def create_demo(model_path=None):
    processor = StyleGAN3Demo(model_path)
    
    with gr.Blocks(title="StyleGAN3-Inspired Image Processor") as demo:
        gr.Markdown("""
            # Fast StyleGAN3-Inspired Image Processor
            Process images while preserving translation and rotation properties.
            
            ## Tips for Best Performance:
            - Start with smaller image sizes (64-128px) and increase if needed
            - Use CPU mode if GPU memory issues occur
            - Lower GPU memory usage for more stability
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=256
                )
                
                with gr.Row():
                    target_size = gr.Slider(
                        minimum=32, 
                        maximum=1024,  # Increased maximum
                        value=256,     # Higher default
                        step=64,       # Larger steps
                        label="Target Size",
                        info="Higher = More detail, but slower"
                    )
                
                with gr.Row():
                    filter_strength = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.5, 
                        step=0.1,
                        label="Filter Strength",
                        info="Higher = Smoother output"
                    )
                
                with gr.Accordion("Color & Detail Settings", open=True):
                    with gr.Row():
                        hue_shift = gr.Slider(
                            minimum=-1.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.01,
                            label="Hue Shift",
                            info="Adjust color hue (-1.0 to 1.0)"
                        )
                        
                        saturation = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.05,
                            label="Saturation",
                            info="Color intensity"
                        )
                    
                    with gr.Row():
                        contrast = gr.Slider(
                            minimum=0.5,
                            maximum=3.0,
                            value=1.2,
                            step=0.1,
                            label="Contrast",
                            info="Image contrast (higher = punchier)"
                        )
                        
                        brightness = gr.Slider(
                            minimum=-0.5,
                            maximum=0.5,
                            value=0.0,
                            step=0.05,
                            label="Brightness",
                            info="Overall brightness adjustment"
                        )
                    
                    with gr.Row():
                        shadows = gr.Slider(
                            minimum=-0.5,
                            maximum=0.5,
                            value=0.0,
                            step=0.05,
                            label="Shadow Lift",
                            info="Adjust shadow levels"
                        )
                        
                        highlights = gr.Slider(
                            minimum=-0.5,
                            maximum=0.5,
                            value=0.0,
                            step=0.05,
                            label="Highlight Control",
                            info="Adjust highlight levels"
                        )
                
                with gr.Accordion("Processing Settings", open=False):
                    with gr.Row():
                        memory_usage = gr.Slider(
                            minimum=0.1, 
                            maximum=0.3, 
                            value=0.2, 
                            step=0.05,
                            label="GPU Memory Limit",
                            info="Lower = More stable"
                        )
                        
                        fourier_scale = gr.Slider(
                            minimum=0.01,    # Much lower minimum for maximum detail
                            maximum=10.0,    # Lower maximum to prevent over-smoothing
                            value=2.0,      # Default to more detail
                            step=0.01,      # Finer control
                            label="Detail Level",
                            info="Lower = More detail preservation (0.01-1.0 for maximum detail)"
                        )
                    
                    use_cpu = gr.Checkbox(
                        label="CPU Mode",
                        info="Check if GPU runs out of memory",
                        value=False
                    )
                    
                    save_images = gr.Checkbox(
                        label="Save Results",
                        info="Save input/output pairs",
                        value=False
                    )
                
                process_btn = gr.Button(
                    "Process Image", 
                    variant="primary",
                    scale=2
                )
                
            with gr.Column():
                output_image = gr.Image(
                    label="Processed Image",
                    height=256
                )
                status = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        with gr.Row():
            hue_shift = gr.Slider(
                minimum=-0.5,
                maximum=0.5,
                value=0.0,
                step=0.01,
                label="Hue Shift",
                info="Adjust color hue (-0.5 to 0.5)"
            )
            
            contrast = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Contrast",
                info="Adjust image contrast"
            )
        
        def process_with_status(image, target_size, filter_strength, fourier_scale, 
                              save_images, memory_frac, use_cpu, hue_shift, contrast,
                              saturation, brightness, shadows, highlights):
            try:
                # Clear GPU memory first
                if not use_cpu:
                    torch.cuda.empty_cache()
                    torch.cuda.set_per_process_memory_fraction(memory_frac)
                
                # Switch device if needed
                if use_cpu and processor.device.type != 'cpu':
                    processor.model = processor.model.cpu()
                    processor.device = torch.device('cpu')
                elif not use_cpu and processor.device.type != 'cuda':
                    processor.model = processor.model.cuda()
                    processor.device = torch.device('cuda')
                
                # Process image with progress updates
                processed = processor.process_image(
                    image, 
                    target_size=target_size,
                    filter_strength=filter_strength, 
                    fourier_scale=fourier_scale,
                    hue_shift=hue_shift,
                    contrast=contrast,
                    saturation=saturation,
                    brightness=brightness,
                    shadows=shadows,
                    highlights=highlights
                )
                
                if save_images and processed is not None:
                    processor.save_processed_pair(image, processed)
                
                return processed, "Processing completed successfully"
            except Exception as e:
                return None, f"Error: {str(e)}"
        
        process_btn.click(
            fn=process_with_status,
            inputs=[
                input_image, target_size, filter_strength, fourier_scale, 
                save_images, memory_usage, use_cpu, hue_shift, contrast,
                saturation, brightness, shadows, highlights
            ],
            outputs=[output_image, status]
        )
    
    # Launch with optimized settings for fast inference
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True,
        max_threads=1,  # Limit concurrent processing
        quiet=True      # Reduce logging overhead
    )


if __name__ == "__main__":
    try:
        # Initialize CUDA memory management
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
        
        # Launch demo
        create_demo(model_path=None)
        
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nError: {str(e)}")
        torch.cuda.empty_cache()
