# FrequencyNet: A Dual-Domain Neural Architecture for Image Processing

FrequencyNet is an advanced hybrid neural network architecture that combines spatial and frequency domain processing to enhance image translation tasks. Originally inspired by StyleGAN3's alias-free design principles, FrequencyNet goes beyond traditional preprocessing by implementing a novel dual-domain approach.

## Key Features

- **Dual-Domain Processing**: Simultaneously operates in both spatial and frequency domains for optimal signal preservation
- **Adaptive Frequency Masking**: Implements separate processing paths for luminance and chrominance channels
- **Signal Continuity**: Uses careful filtering to maintain signal continuity throughout the network
- **Translation & Rotation Equivariance**: Ensures features transform consistently with image content
- **Lightweight Architecture**: Only ~100K parameters, making it suitable for real-time processing
- **Hybrid Neural Network**: Combines traditional CNNs with Fourier domain processing

## Architecture

FrequencyNet consists of three main components:

1. **Spatial Processing Network**:
   - Input layer: 3-channel RGB image
   - Two convolutional layers (64 channels each) with LeakyReLU activation
   - Output layer: 3-channel processed image

2. **Frequency Domain Processor**:
   - Channel-wise 2D Fourier transforms
   - Adaptive frequency masking for luminance and chrominance
   - Phase information preservation for spatial coherence

3. **Signal Blending Module**:
   - Learnable weighted combination of spatial and frequency features
   - Output = α·Spatial_out + (1-α)·Frequency_out
   - α empirically set to 0.7 based on validation experiments

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- PIL (Pillow)
- numpy
- gradio (for the interactive demo)
- CycleGAN repository (from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/stylegan3-cyclegan-processor.git
cd stylegan3-cyclegan-processor
```

2. Clone the CycleGAN repository:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
```

3. Install dependencies:
```bash
pip install torch torchvision pillow numpy gradio
```

## Usage

### Pre-processing Images

The FrequencyNet pre-processor enhances images before feeding them to CycleGAN, making the translation process more coherent and natural.

```bash
python stylegan3_pre_processor.py --mode preprocess \
    --input_dir path/to/input/images \
    --output_dir path/to/output/images \
    --batch_size 4
```

### Preparing a Dataset for CycleGAN

To prepare a complete dataset (with both domains) for CycleGAN training:

```bash
python cyclegan_integration.py --mode prepare \
    --input_dir path/to/original/dataset \
    --dataset_name your_dataset_name
```

The input directory should have the standard CycleGAN structure with `trainA` and `trainB` subdirectories.

### Processing and Training in One Step

To process your images and train CycleGAN in a single command:

```bash
python cyclegan_integration.py --mode train \
    --input_dir path/to/original/dataset \
    --dataset_name your_dataset_name \
    --cyclegan_path ./pytorch-CycleGAN-and-pix2pix \
    --epochs 200 \
    --batch_size 1
```

### Training the Preprocessor (Optional)

For better results, you can train the pre-processor on your specific domain:

```bash
python stylegan3_pre_processor.py --mode train \
    --train_dir path/to/training/images \
    --val_dir path/to/validation/images \
    --output_model_path path/to/save/model.pth \
    --epochs 10 \
    --batch_size 4 \
    --lr 0.0002
```

### Interactive Demo

The project includes a Gradio-based interactive demo that allows you to process images in real-time:

```bash
python gradio_demo.py
```

This launches a web interface where you can upload an image and see the FrequencyNet-enhanced version.

## How It Works

The FrequencyNet pre-processor:

1. **Applies low-pass filtering** before and after operations to prevent aliasing
2. **Integrates Fourier features** to enhance spatial awareness
3. **Uses filtered nonlinearities** to maintain signal continuity
4. **Incorporates self-attention** with alias-free processing
5. **Preserves equivariance properties** during transformations

## Integration with CycleGAN Workflow

1. Pre-process your source domain images using this preprocessor
2. Pre-process your target domain images (optional but recommended)
3. Use the processed images for CycleGAN training
4. The resulting CycleGAN model will benefit from improved translation coherence and better handling of spatial transformations

## Tips for Best Results

- For optimal results, train the pre-processor on your specific domain before using it
- Use a consistent image resolution throughout your workflow
- The preprocessor works best with clean, well-lit images
- Experiment with different filter sizes and cutoff frequencies for your specific use case

## Acknowledgments

- This implementation is inspired by the [StyleGAN3 paper](https://nvlabs.github.io/stylegan3/) by Karras et al. and the [official StyleGAN3 implementation](https://github.com/NVlabs/stylegan3)
- CycleGAN implementation from [Zhu et al.](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Citation

If you use this model in your research, please cite our work and the original StyleGAN3 paper:

```bibtex
@inproceedings{Karras2021,
  author = {Tero Karras and Miika Aittala and Samuli Laine and Erik H\"ark\"onen and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  title = {Alias-Free Generative Adversarial Networks},
  booktitle = {Proc. NeurIPS},
  year = {2021}
}
```

StyleGAN3 official implementation: [https://github.com/NVlabs/stylegan3](https://github.com/NVlabs/stylegan3)
