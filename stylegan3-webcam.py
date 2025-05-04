import gradio as gr
import cv2
import numpy as np
import torch
import os
import pyvirtualcam
from pyvirtualcam import PixelFormat
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Import the StyleGAN3 preprocessor
from stylegan3_pre_processor import StyleGAN3PreProcessor

# Device selection with MPS support
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Set memory management settings for CUDA
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    MEMORY_FRACTION = 0.6
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)

# Global variables for model management
current_stylegan_model = None

def load_stylegan_model():
    """Load the StyleGAN3 preprocessor model"""
    global current_stylegan_model
    if current_stylegan_model is None:
        model = StyleGAN3PreProcessor()
        model = model.to(DEVICE).eval()
        current_stylegan_model = model
    return current_stylegan_model

def rgb_to_hsv(rgb):
    # Convert RGB to HSV (numpy implementation)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    r, g, b = r/255.0, g/255.0, b/255.0
    
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    
    deltac = maxc - minc
    s = np.zeros_like(maxc)
    s[maxc != 0] = deltac[maxc != 0] / maxc[maxc != 0]
    
    h = np.zeros_like(deltac)
    
    # Mask for each case
    mask_r_max = (maxc == r)
    mask_g_max = (maxc == g)
    mask_b_max = (maxc == b)
    
    h[mask_r_max] = ((g[mask_r_max] - b[mask_r_max]) / deltac[mask_r_max]) % 6
    h[mask_g_max] = ((b[mask_g_max] - r[mask_g_max]) / deltac[mask_g_max]) + 2
    h[mask_b_max] = ((r[mask_b_max] - g[mask_b_max]) / deltac[mask_b_max]) + 4
    
    h[deltac == 0] = 0
    h = h / 6.0
    
    return np.stack([h, s, v], axis=2)

def hsv_to_rgb(hsv):
    # Convert HSV to RGB (numpy implementation)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    i = np.floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    i = i.astype(np.int32) % 6
    
    rgb = np.zeros_like(hsv)
    mask = (i == 0)
    rgb[mask, 0] = v[mask]
    rgb[mask, 1] = t[mask]
    rgb[mask, 2] = p[mask]
    
    mask = (i == 1)
    rgb[mask, 0] = q[mask]
    rgb[mask, 1] = v[mask]
    rgb[mask, 2] = p[mask]
    
    mask = (i == 2)
    rgb[mask, 0] = p[mask]
    rgb[mask, 1] = v[mask]
    rgb[mask, 2] = t[mask]
    
    mask = (i == 3)
    rgb[mask, 0] = p[mask]
    rgb[mask, 1] = q[mask]
    rgb[mask, 2] = v[mask]
    
    mask = (i == 4)
    rgb[mask, 0] = t[mask]
    rgb[mask, 1] = p[mask]
    rgb[mask, 2] = v[mask]
    
    mask = (i == 5)
    rgb[mask, 0] = v[mask]
    rgb[mask, 1] = p[mask]
    rgb[mask, 2] = q[mask]
    
    rgb = (rgb * 255).astype(np.uint8)
    return rgb

def adjust_colors(image, hue_shift=0.0, contrast=1.2, saturation=1.0, 
                brightness=0.0, shadows=0.0, highlights=0.0):
    """Advanced color adjustment with better contrast and brightness control"""
    try:
        # Convert to float
        img = image.astype(np.float32) / 255.0
        
        # Apply contrast first (more aggressive)
        if contrast != 1.0:
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            img = np.sign(img - mean) * (np.abs(img - mean) ** (1/contrast)) + mean
        
        # Apply brightness
        if brightness != 0:
            img = img + brightness
        
        # Adjust shadows and highlights more aggressively
        if shadows != 0 or highlights != 0:
            luminance = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
            shadow_mask = np.power(1.0 - luminance, 2)  # More aggressive shadow adjustment
            highlight_mask = np.power(luminance, 2)     # More aggressive highlight adjustment
            
            for i in range(3):
                img[:,:,i] = img[:,:,i] + (shadows * 1.5) * shadow_mask
                img[:,:,i] = img[:,:,i] + (highlights * 1.5) * highlight_mask
        
        # Convert to HSV for hue and saturation
        hsv = rgb_to_hsv(img * 255)
        
        # Adjust hue
        hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 1.0
        
        # More aggressive saturation
        if saturation != 1.0:
            hsv[:,:,1] = np.clip(hsv[:,:,1] * (saturation * 1.2), 0, 1)
        
        # Convert back to RGB
        rgb = hsv_to_rgb(hsv)
        
        return rgb
        
    except Exception as e:
        print(f"Error adjusting colors: {str(e)}")
        return image

def preprocess_for_stylegan(image, target_size=512):
    """Preprocess image for StyleGAN3 input"""
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
    
    # Save original dimensions
    original_size = (image_pil.width, image_pil.height)
    
    # Calculate new dimensions
    aspect_ratio = original_size[0] / original_size[1]
    
    if original_size[0] > original_size[1]:
        new_width = min(target_size, 2048)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(target_size, 2048)
        new_width = int(new_height * aspect_ratio)
    
    new_width = max(new_width, 32)
    new_height = max(new_height, 32)
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((new_height, new_width), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Process image
    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    return input_tensor, original_size

def postprocess_from_stylegan(tensor, original_size):
    """Convert StyleGAN3 output tensor to numpy image with original dimensions"""
    tensor = tensor.squeeze(0).cpu()
    tensor = ((tensor * 0.5) + 0.5).clamp(0, 1)
    tensor = tensor.permute(1, 2, 0).numpy() * 255
    
    # Convert to uint8
    image = tensor.astype(np.uint8)
    
    # Resize back to original dimensions
    image = cv2.resize(image, original_size)
    return image

@torch.inference_mode()
def process_image_with_stylegan(image, target_size=512, filter_strength=0.75, 
                              fourier_scale=2.0, hue_shift=0.0, contrast=1.2, 
                              saturation=1.0, brightness=0.0, shadows=0.0, 
                              highlights=0.0):
    """Process an image with StyleGAN3 preprocessor"""
    model = load_stylegan_model()
    
    # Save original dimensions
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    
    # Preprocess
    input_tensor, _ = preprocess_for_stylegan(image, target_size)
    
    # Process image through StyleGAN3
    with torch.no_grad():
        try:
            output = model(input_tensor)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU memory exceeded, falling back to CPU...")
                torch.cuda.empty_cache()
                model = model.cpu()
                output = model(input_tensor.cpu())
                model = model.to(DEVICE)
            else:
                raise e
    
    # Postprocess
    output_np = postprocess_from_stylegan(output, original_size)
    
    # Apply color adjustments
    output_np = adjust_colors(
        output_np,
        hue_shift=hue_shift,
        contrast=contrast,
        saturation=saturation,
        brightness=brightness,
        shadows=shadows,
        highlights=highlights
    )
    
    return output_np

def blend_images(original, processed, blend_alpha=0.3):
    """
    Blend the original image with the processed image
    
    Parameters:
    - original: Original webcam frame
    - processed: StyleGAN3 processed frame
    - blend_alpha: Blend strength (0.0 = processed only, 1.0 = original only)
    """
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Blend using weighted average
    result = cv2.addWeighted(processed, 1.0 - blend_alpha, original, blend_alpha, 0)
    
    return result

def get_screen_dimensions():
    """Get the dimensions of the primary screen"""
    try:
        import screeninfo
        screen = screeninfo.get_monitors()[0]
        return screen.width, screen.height
    except:
        # Default resolution if screeninfo not available
        print("Could not detect screen dimensions, using defaults")
        return 1920, 1080

def process_webcam_with_stylegan(blend_alpha=0.3, target_size=512, 
                                filter_strength=0.75, fourier_scale=2.0,
                                hue_shift=0.0, contrast=1.2, saturation=1.0,
                                brightness=0.0, shadows=0.0, highlights=0.0,
                                source_type="webcam"):
    """Process webcam or desktop with StyleGAN3 preprocessor"""
    
    if source_type == "webcam":
        # Open the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return "Error: Could not open webcam"
        
        # Read a test frame to get dimensions
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam")
            return "Error: Could not read from webcam"
        
        # Get frame dimensions
        frame_height, frame_width = test_frame.shape[:2]
    else:  # Desktop capture
        try:
            # Get screen dimensions
            screen_width, screen_height = get_screen_dimensions()
            
            # For desktop capturing
            if os.name == 'nt':  # Windows
                from PIL import ImageGrab
                # Take a test capture to verify
                test_frame = np.array(ImageGrab.grab(bbox=(0, 0, screen_width, screen_height)))
                test_frame = cv2.cvtColor(test_frame, cv2.COLOR_RGB2BGR)
            else:  # Linux/Mac
                try:
                    import mss
                    sct = mss.mss()
                    monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
                    test_frame = np.array(sct.grab(monitor))
                    test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGRA2BGR)
                except ImportError:
                    print("Error: Could not import mss for screen capture. Please install it with 'pip install mss'")
                    return "Error: Missing mss library for screen capture"
            
            # Get frame dimensions
            frame_height, frame_width = test_frame.shape[:2]
            print(f"Screen capture dimensions: {frame_width}x{frame_height}")
        except Exception as e:
            print(f"Error initializing screen capture: {str(e)}")
            return f"Error initializing screen capture: {str(e)}"
    
    # Get frame dimensions
    frame_height, frame_width = test_frame.shape[:2]
    print(f"Webcam frame dimensions: {frame_width}x{frame_height}")
    
    # Create a preview window
    preview_window = "StyleGAN3 Preprocessor Preview"
    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
    
    try:
        # Initialize virtual camera
        with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=30, 
                                fmt=PixelFormat.BGR, backend='obs') as cam:
            print(f'Using virtual camera: {cam.device}')
            print(f'Virtual camera dimensions: {cam.width}x{cam.height}')
            
            frame_count = 0
            
            # Pre-load model
            load_stylegan_model()
            
            while True:
                # Capture frame
                if source_type == "webcam":
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:  # Desktop capture
                    try:
                        if os.name == 'nt':  # Windows
                            from PIL import ImageGrab
                            frame = np.array(ImageGrab.grab(bbox=(0, 0, screen_width, screen_height)))
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        else:  # Linux/Mac
                            frame = np.array(sct.grab(monitor))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    except Exception as e:
                        print(f"Error capturing screen: {str(e)}")
                        break
                
                # Print dimensions occasionally for debugging
                if frame_count % 100 == 0:
                    print(f"Frame {frame_count} dimensions: {frame.shape}")
                frame_count += 1
                
                # Convert BGR to RGB for StyleGAN3
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with StyleGAN3
                processed = process_image_with_stylegan(
                    frame_rgb,
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
                
                # Convert RGB back to BGR for display and virtual cam
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                
                # Blend with original
                if blend_alpha > 0:
                    output = blend_images(frame, processed_bgr, blend_alpha)
                else:
                    output = processed_bgr
                
                # Ensure output has the exact dimensions expected by the virtual camera
                if output.shape[0] != frame_height or output.shape[1] != frame_width:
                    output = cv2.resize(output, (frame_width, frame_height))
                
                # Show preview
                cv2.imshow(preview_window, output)
                
                # Send to virtual camera
                try:
                    cam.send(output)
                    cam.sleep_until_next_frame()
                except Exception as e:
                    print(f"Error sending to virtual camera: {e}")
                
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except Exception as e:
        print(f"Error in webcam processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error in webcam processing: {str(e)}"
    
    finally:
        # Clean up
        if source_type == "webcam":
            cap.release()
        cv2.destroyAllWindows()
    
    return "Processing completed"

###########################################
# Gradio Interface
###########################################

with gr.Blocks(title="StyleGAN3 Webcam/Desktop Processor") as demo:
    gr.Markdown("# StyleGAN3-Inspired Video Processor")
    
    with gr.Row():
        with gr.Column():
            source_type = gr.Radio(
                choices=["webcam", "desktop"],
                value="webcam",
                label="Video Source"
            )
            
            blend_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.1,
                label="Original Video Blend Strength (0 = StyleGAN3 only, 1 = original only)"
            )
            
            target_size = gr.Slider(
                minimum=128,
                maximum=1024,
                value=512,
                step=64,
                label="Target Processing Size (higher = more detail but slower)"
            )
            
            fourier_scale = gr.Slider(
                minimum=0.01,
                maximum=10.0,
                value=2.0,
                step=0.1,
                label="Detail Level (lower = more detail preservation)"
            )
            
            with gr.Accordion("Color Adjustments", open=True):
                with gr.Row():
                    hue_shift = gr.Slider(
                        minimum=-0.5,
                        maximum=0.5,
                        value=0.0,
                        step=0.05,
                        label="Hue Shift"
                    )
                    
                    saturation = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Saturation"
                    )
                
                with gr.Row():
                    contrast = gr.Slider(
                        minimum=0.5,
                        maximum=3.0,
                        value=1.2,
                        step=0.1,
                        label="Contrast"
                    )
                    
                    brightness = gr.Slider(
                        minimum=-0.5,
                        maximum=0.5,
                        value=0.0,
                        step=0.05,
                        label="Brightness"
                    )
                
                with gr.Row():
                    shadows = gr.Slider(
                        minimum=-0.5,
                        maximum=0.5,
                        value=0.0,
                        step=0.05,
                        label="Shadow Lift"
                    )
                    
                    highlights = gr.Slider(
                        minimum=-0.5,
                        maximum=0.5,
                        value=0.0,
                        step=0.05,
                        label="Highlight Control"
                    )
            
            start_button = gr.Button("Start Processing", variant="primary")
        
        with gr.Column():
            output_status = gr.Textbox(
                label="Status",
                value="Ready to start...",
                interactive=False
            )
    
    # Instructions
    gr.Markdown("""
    ### Instructions:
    1. Choose your video source (webcam or desktop screen)
    2. Adjust the blend strength between the original feed and the StyleGAN3 processed feed
    3. Set the target processing size (higher quality but slower processing)
    4. Adjust color settings to get the desired look
    5. Click "Start Processing" to begin the virtual camera feed
    6. A preview window will open - press 'q' in that window to stop processing
    
    **Requirements:**
    - For webcam: A working webcam
    - For desktop capture: 
      - Windows: No additional requirements
      - macOS/Linux: Install mss library (`pip install mss`)
    - Virtual camera: pyvirtualcam and a virtual camera device (like OBS Virtual Camera)
    """)
    
    def start_processing(source, blend_alpha, target_size, fourier_scale, 
                        hue_shift, contrast, saturation,
                        brightness, shadows, highlights):
        try:
            result = process_webcam_with_stylegan(
                blend_alpha=blend_alpha,
                target_size=target_size,
                fourier_scale=fourier_scale,
                hue_shift=hue_shift,
                contrast=contrast,
                saturation=saturation,
                brightness=brightness,
                shadows=shadows,
                highlights=highlights,
                source_type=source
            )
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    start_button.click(
        fn=start_processing,
        inputs=[
            source_type, blend_slider, target_size, fourier_scale, 
            hue_shift, contrast, saturation,
            brightness, shadows, highlights
        ],
        outputs=output_status
    )

if __name__ == "__main__":
    demo.launch()