# shecodes_iitj_myntra

# Stable Diffusion 3 with DreamBooth LoRA

This repository contains code for training and inference using Stable Diffusion 3 with DreamBooth and LoRA (Low-Rank Adaptation). The project is designed to generate images based on text prompts and fine-tune models with specific datasets.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Computing Embeddings](#computing-embeddings)
  - [Training](#training)
  - [Inference](#inference)
- [Files](#files)
- [Requirements](#requirements)

## Introduction

This project leverages the power of Stable Diffusion 3, DreamBooth, and LoRA to generate high-quality images from text prompts. The code is divided into three main scripts: `compute_embeddings.py`, `train.py`, and `sd3_dreambooth_lora.py`. Additionally, a `requirements.txt` file is provided to set up the necessary dependencies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Computing Embeddings

The `compute_embeddings.py` script computes the embeddings for a given set of images and saves them in a Parquet file.

1. Ensure your images are stored in a directory (e.g., `outfits`) with each image accompanied by a text file containing the caption (e.g., `image1.jpg` and `image1.txt`).

#### Arguments

- `--prompt`: The instance prompt (default: `"photos of trendy genz outfits"`).
- `--max_sequence_length`: Maximum sequence length for computing embeddings (default: `77`).
- `--local_data_dir`: Path to the directory containing instance images (default: `"outfits"`).
- `--output_path`: Path to save the Parquet file with embeddings (default: `"sample_embeddings.parquet"`).

### Training

The `train.py` script trains the model using the computed embeddings and saves the trained model.

1. Ensure your instance data and embeddings file (`sample_embeddings.parquet`) are available.

#### Arguments

- `--pretrained_model_name_or_path`: Path to the pretrained model or model identifier from Hugging Face.
- `--instance_data_dir`: Path to the directory containing instance images.
- `--data_df_path`: Path to the Parquet file with embeddings.
- `--output_dir`: Directory to save the trained model.
- `--mixed_precision`: Mixed precision mode (e.g., `fp16`).
- `--instance_prompt`: Instance prompt for training.
- Additional arguments are available for customization (see script for full list).

### Inference

The `sd3_dreambooth_lora.py` script handles dependencies installation, Hugging Face login,training and inference.

1. Ensure you have the API key for Hugging Face login.

2. Run the inference script:
   ```bash
   python sd3_dreambooth_lora.py
   ```

#### Functions

- `install_dependencies()`: Installs necessary dependencies.
- `huggingface_login(api_key)`: Logs into Hugging Face using the provided API key.
- `clone_diffusers_repo()`: Clones the Diffusers repository.
- `download_instance_data()`: Downloads instance data.
- `compute_embeddings()`: Computes embeddings.
- `flush_memory()`: Clears memory.
- `train_model()`: Trains the model.
- `run_inference()`: Runs inference and saves the generated image.

## Files

- `compute_embeddings.py`: Script to compute embeddings.
- `train.py`: Script to train the model.
- `sd3_dreambooth_lora.py`: Script for dependencies installation, Hugging Face login,training and inference.
- `requirements.txt`: List of required Python packages.

Install the requirements using:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the Apache 2.0 License.
```

You can copy this entire file into your README.md file in your GitHub repository.

