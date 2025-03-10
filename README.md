# PyTorch PaliGemma

A PyTorch implementation of Google's PaliGemma multimodal vision-language model. This repository provides a lightweight implementation for running inference with PaliGemma models.

## Overview

PaliGemma is a multimodal vision-language model that combines:
- A SigLIP vision encoder (based on Vision Transformer architecture)
- A Gemma language model

The model can process images along with text prompts to generate text responses based on visual content.

## Key Technical Features

### Vision Encoder
- **Vision Transformer Architecture**: Implements SigLIP (Simple Gated CLIP) vision encoder with patch-based image processing
- **Rotary Positional Embedding**: Enhances the model's understanding of spatial relationships in images
- **Multi-Head Attention**: Enables the model to focus on different parts of the image simultaneously

### Language Model
- **Gemma Architecture**: Transformer-based language model with self-attention mechanisms
- **Grouped Query Attention**: Implements efficient attention where multiple query heads share the same key-value pairs, reducing computational overhead
- **RMS Normalization**: Used for stable training and inference

### Multimodal Integration
- **Contrastive Learning**: Implements CLIP/SigLIP-style contrastive learning with Softmax and Cross-Entropy Loss for stability
- **Projection Layer**: Connects vision and language models through a projection layer that maps visual features to the language model's embedding space

### Efficient Inference
- **KV-Cache Implementation**: Caches key-value pairs during autoregressive generation to avoid redundant computations
- **Attention Masks**: Properly handles attention masking for efficient processing
- **Weight Tying**: Shares weights between embedding and output layers to reduce model size
- **Top-P Sampling**: Implements nucleus sampling for more diverse and natural text generation

## Requirements

The project requires the following dependencies:
```
fire==0.6.0
numpy==1.26.4
pillow==10.3.0
safetensors==0.4.3
tokenizers==0.19.1
torch==2.3.0
torchaudio==2.3.0
torchvision==0.18.0
tqdm==4.66.4
transformers==4.41.2
```

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running Inference

You can run inference using the provided `inference.py` script:

```python
python inference.py \
    --model_path "/path/to/paligemma-model" \
    --prompt "Describe this image:" \
    --image_file_path "path/to/image.jpg" \
    --max_tokens_to_generate 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --do_sample False \
    --only_cpu False
```

Alternatively, you can use the provided shell script:

```bash
./launch_inference.sh
```

Make sure to update the parameters in the script to match your setup.

### Parameters

- `model_path`: Path to the PaliGemma model weights
- `prompt`: Text prompt to condition the model
- `image_file_path`: Path to the input image
- `max_tokens_to_generate`: Maximum number of tokens to generate
- `temperature`: Temperature for sampling (higher = more random)
- `top_p`: Top-p sampling parameter (nucleus sampling)
- `do_sample`: Whether to use sampling or greedy decoding
- `only_cpu`: Force CPU-only inference

## Model Architecture Details

### 1. SigLIP Vision Model (`modeling_siglip.py`)
- **Vision Transformer**: Processes images as sequences of patches
- **Patch Embedding**: Converts image patches to embeddings using convolutional layers
- **Position Embedding**: Adds positional information to patch embeddings
- **Self-Attention Layers**: Process visual information with multi-head attention
- **Layer Normalization**: Stabilizes training and inference

### 2. Gemma Language Model (`modeling_gemma.py`)
- **Transformer Architecture**: Processes text with self-attention mechanisms
- **Rotary Positional Embedding**: Enhances the model's understanding of token positions
- **Grouped Query Attention**: Multiple query heads share the same key-value pairs for efficiency
- **RMS Normalization**: Used instead of Layer Normalization for better stability
- **MLP Blocks**: Process information between attention layers
- **KV-Cache**: Efficiently stores key-value pairs for faster autoregressive generation

### 3. PaliGemma Processor (`processing_paligemma.py`)
- **Image Processing**: Resizes, normalizes, and prepares images for the vision encoder
- **Text Processing**: Tokenizes text and adds special tokens
- **Multimodal Integration**: Combines image and text inputs for the model
- **Special Token Handling**: Manages image tokens and other special tokens

### 4. Inference Pipeline (`inference.py`)
- **Input Processing**: Prepares images and text for the model
- **Autoregressive Generation**: Generates text tokens one by one
- **KV-Cache Management**: Efficiently manages the key-value cache during generation
- **Top-P Sampling**: Implements nucleus sampling for text generation

## Implementation Highlights

1. **Efficient Attention Mechanism**:
   - Implements grouped query attention where multiple query heads share the same key-value pairs
   - Reduces computational overhead while maintaining model quality

2. **Rotary Positional Embedding**:
   - Implements RoPE (Rotary Position Embedding) for both vision and language components
   - Provides better positional understanding compared to absolute positional embeddings

3. **Contrastive Learning**:
   - Implements CLIP/SigLIP-style contrastive learning approach
   - Uses Softmax and Cross-Entropy Loss for stable training

4. **Optimized Generation**:
   - KV-Cache implementation for faster autoregressive generation
   - Proper attention masking for efficient processing
   - Weight tying between embedding and output layers
   - Top-P sampling for diverse and natural text generation

## Notes

The repository includes additional documentation in the `notes/` directory:
- From CLIP to SigLIP.pdf
- KV-Cache.pdf
- Multi-Head Attention.pdf
- Normalization.pdf

## License

Please refer to the original PaliGemma model license for usage restrictions.

## Acknowledgements

This implementation is based on Google's PaliGemma model. For more information, see the [official PaliGemma blog post](https://huggingface.co/blog/paligemma).