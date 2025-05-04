# StyleGAN3-Inspired Pre-Processor for CycleGAN

This project implements image processing techniques inspired by StyleGAN3's alias-free design principles to enhance image translation using CycleGAN. The pre-processor adds translation and rotation equivariance to images, which helps CycleGAN generate more consistent and natural transformations.

## Features

- **Signal Continuity**: Uses careful filtering to maintain signal continuity throughout the network, preventing aliasing artifacts
- **Translation Equivariance**: Ensures features transform consistently with the image content, instead of being "glued" to pixel coordinates
- **Rotation Awareness**: Enhanced ability to handle rotations in a more equivariant manner
- **Fourier Features**: Uses continuous Fourier features to better represent spatial relationships
- **Self-Attention**: Incorporates alias-free self-attention to better understand global image context

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

The StyleGAN3 pre-processor enhances images before feeding them to CycleGAN, making the translation process more coherent and natural.

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

This launches a web interface where you can upload an image and see the StyleGAN3-enhanced version.

## How It Works

The StyleGAN3-inspired pre-processor:

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

- This implementation is inspired by the [StyleGAN3 paper](https://nvlabs.github.io/stylegan3/) by Karras et al.
- CycleGAN implementation from [Zhu et al.](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
