# Image-Caption-Model-with-Semantic-Alignment

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
  - [SEVE (Semantic-Enhanced Visual Encoder)](#seve-semantic-enhanced-visual-encoder)
  - [QSB (Query-based Semantic Bridge)](#qsb-query-based-semantic-bridge)
  - [DLA (Dynamic LoRA Adaptation)](#dla-dynamic-lora-adaptation)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [Base Model Training](#base-model-training)
  - [Training with QSB](#training-with-qsb)
  - [Training with DLA](#training-with-dla)
  - [Combined Training (SEVE + QSB + DLA)](#combined-training-seve--qsb--dla)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Overview

**Image-Caption-Model-with-Semantic-Alignment** is an advanced image captioning system that leverages three novel semantic alignment techniques to bridge the gap between visual features and natural language generation. Built on top of CLIP vision encoders and GPT-2 language models, this project introduces innovative approaches to enhance cross-modal understanding and generate more accurate, contextually relevant image descriptions.

The model achieves state-of-the-art performance on COCO and NoCaps benchmarks through its unique combination of semantic enhancement, query-based bridging, and parameter-efficient adaptation mechanisms.
![](./pic/清晰.png)
### Key Contributions
- **SEVE**: Enhances visual feature representation for better semantic alignment
- **QSB**: Introduces learnable query tokens for improved vision-language fusion
- **DLA**: Enables efficient fine-tuning through dynamic low-rank adaptation


## Key Features

✅ **Three Novel Semantic Alignment Techniques**
- SEVE for enhanced visual encoding
- QSB for query-based semantic bridging
- DLA for parameter-efficient adaptation

✅ **Multiple Architecture Configurations**
- MLP-based mapping networks
- Transformer-based mapping networks
- Hybrid architectures combining multiple techniques

✅ **Comprehensive Dataset Support**
- COCO (Common Objects in Context)
- Flickr30k
- Conceptual Captions
- NoCaps (for zero-shot evaluation)

✅ **Advanced Generation Strategies**
- Beam search decoding
- Nucleus sampling (top-p)
- Temperature-controlled generation
- Length-constrained decoding optimized for CIDEr

✅ **Efficient Training Pipeline**
- Gradient checkpointing support
- Mixed precision training ready
- Distributed training compatible
- Comprehensive logging and monitoring

---

## Architecture

The model follows an encoder-decoder architecture with three key innovations:

### SEVE (Semantic-Enhanced Visual Encoder)

SEVE enhances the standard CLIP visual encoder by introducing additional semantic processing layers that better capture object relationships and scene context.

**Key Components:**
- **Base Encoder**: CLIP ViT-B/16 for initial feature extraction
- **Semantic Enhancement Layer**: Additional transformation layers that enrich visual features with contextual information
- **Normalization Module**: L2 normalization for stable training and better gradient flow

**Implementation Details:**
```python
prefix = clip_model.encode_image(image)
prefix = prefix / prefix.norm(2, -1)
prefix_embed = model.clip_project(prefix)
```
**Benefits:**
- Improved object-level semantic understanding
- Better handling of complex scenes
- Enhanced feature discriminability

### QSB (Query-based Semantic Bridge)

QSB introduces learnable query tokens inspired by the Q-Former architecture from BLIP-2, creating a flexible bridge between visual features and language representations.

**Architecture:**
- **Query Tokens**: 32 learnable query vectors that interact with visual features
- **Cross-Attention Mechanism**: Queries attend to visual features to extract relevant semantic information
- **Projection Layer**: Maps query outputs to GPT-2 embedding space (768 dimensions)

**Technical Specifications:**
class AdaptiveQFormer(nn.Module): - Input dimension: 512 (CLIP ViT-B/16) or 640 (ResNet50x4) - Query tokens: 32 - Output dimension: 768 (GPT-2 hidden size) - Attention heads: Configurable (default: 8) - Layers: Configurable (default: 12)
**Forward Process:**
1. Visual features are layer-normalized
2. Learnable queries attend to visual features via cross-attention
3. Query outputs are projected to language model dimension
4. Result serves as prefix for GPT-2 decoder

**Advantages:**
- Flexible number of output tokens (32 vs fixed prefix length)
- Better semantic compression of visual information
- Improved generalization to unseen concepts

### DLA (Dynamic LoRA Adaptation)

DLA applies Low-Rank Adaptation (LoRA) to the GPT-2 language model, enabling efficient fine-tuning while maintaining pre-trained knowledge.

**Configuration:**
LoraConfig( r=8, # LoRA rank lora_alpha=32, # Scaling factor target_modules=["q_proj", "c_proj"], # Target attention modules lora_dropout=0.1, # Dropout rate bias="none", task_type="CAUSAL_LM" )
**Key Design Choices:**
- **Frozen Base Model**: GPT-2 pretrained weights remain frozen
- **Trainable LoRA Modules**: Only low-rank decomposition matrices are updated
- **Selective Targeting**: Applied to query projection and output projection layers
- **Parameter Efficiency**: Reduces trainable parameters by ~99% compared to full fine-tuning

**Benefits:**
- Dramatically reduced memory requirements
- Faster training convergence
- Prevention of catastrophic forgetting
- Easy multi-task learning capability


## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- PyTorch 1.12+
- 16GB+ GPU memory recommended

### Step-by-Step Setup

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/Image-Caption-Model-with-Semantic-Alignment.git
cd Image-Caption-Model-with-Semantic-Alignment
```
2. **Create conda environment:**
```
conda create -n imgcap python=3.9
conda activate imgcap
```
3. **Install PyTorch:**
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
4. **Install dependencies:**
```
pip install ftfy regex tqdm pip install git+https://github.com/openai/CLIP.git pip install transformers==4.30.0 pip install peft==0.4.0 pip install pycocotools pip install pycocoevalcap pip install scikit-image pip install Pillow pip install datasets pip install requests
```
## Dataset Preparation

### COCO Dataset
### Flickr30k Dataset
### Conceptual Captions
The project supports direct loading from HuggingFace datasets:
### NoCaps Dataset

For zero-shot evaluation:
## Training

### Base Model Training
```
python clip_qf_train.py --data ./data/oscar_split_vit_base_patch16_train.pkl --valdata ./data/oscar_split_vit_base_patch16_val.pkl --mapping_type transformer --num_layers 12 --prefix_length 32 --clip_length 40 --bs 160 --output_prefix vitbase16 --epoch 40 --output_dir ./checkpoints/base/
```
### Training with SEVE

Enable QSB by using the `clip_qf_train.py` script:

```
python clip_qf_train.py --data ./data/oscar_split_ViT-B_16_train.pkl --mapping_type transformer  --num_layers 12 --valdata ./data/oscar_split_ViT-B_16_val.pkl --prefix_length 32 --clip_length 40  --bs 160 --epoch 40 --output_prefix clip --output_dir ./checkpoints/clip
```
### Training with QSB

Enable QSB by using the `clip_qf_train.py` script:
```
python clip_qf_train.py  --data ./data/oscar_split_vit_base_patch16_train.pkl --valdata ./data/oscar_split_vit_base_patch16_val.pkl --epochs 40 --output_prefix qf --output_dir ./checkpoints/qf --bs 128 --prefix_length 32 --use_qformer --freeze_qformer
```
**QSB-specific Parameters:**
- `--use_qformer`: Enable QSB module
- `--freeze_qformer`: Freeze QSB parameters during training
- `--prefix_length 32`: QSB outputs 32 query tokens

### Training with DLA

```
python clip_qf_train.py  --data ./data/oscar_split_vit_base_patch16_train.pkl --num_layers 12 --valdata ./data/oscar_split_vit_base_patch16_val.pkl --epochs 40 --output_prefix ve --output_dir ./checkpoints/ve --bs 160 --prefix_length 32 --use_lora --clip_length 40
```
**DLA-specific Parameters:**
- `--use_lora`: Enable LoRA adaptation
- LoRA configuration is hardcoded in `clip_qf_train.py`:
  - `r=8`, `lora_alpha=32`, `lora_dropout=0.1`


### Combined Training (SEVE + QSB + DLA)

Train the full model with all three techniques:
```
python clip_qf_train.py  --data ./data/oscar_split_ViT-B_16_train.pkl --valdata ./data/oscar_split_ViT-B_16_val.pkl  --epochs 40 --output_prefix clip_qf_ve --output_dir ./checkpoints/clip_qf_ve --bs 128 --prefix_length 32 --use_lora --use_qformer --freeze_qformer --seed 64
```
## Inference

### Single Image Captioning

Generate caption for a single image:
```
python predict_ve.py --run_model run_val --model_path ./checkpoints/clip_qf_ve/clip_qf_ve_best.pt --output_prefix clipqfve --prefix_length 32 --mapping_network none --dataval ./data/oscar_split_ViT-B_16_val.pkl --beam_size 5 --use_lora --use_qformer --freeze_qformer
```

## Evaluation

### Automatic Metrics

The project uses the official COCO evaluation toolkit to compute:

- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram precision metrics
- **METEOR**: Harmonic mean of precision and recall with stemming
- **ROUGE-L**: Longest common subsequence metric
- **CIDEr**: Consensus-based image description evaluation
- **SPICE**: Semantic propositional image-caption evaluation

### Custom Evaluation Script

```
cd out_score
python testscore.py --gt_file ../data/annotations/captions_val2017.json --pred_file ../data/val_out/your_predictions.json

```

## Results
Result on MSCOCO7_val
![](./pic/result.png)
![](./pic/three_loss.png)
All model's loss can be found in folder pic.


## Citation
If you use this work in your research, please cite:
```bibtex
@article{semantic_alignment_captioning, title={Image Captioning with Semantic Alignment: SEVE, QSB, and DLA}, author={Your Name}, journal={arXiv preprint}, year={2026} }
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **CLIP**: OpenAI's Contrastive Language-Image Pre-training
- **GPT-2**: OpenAI's Generative Pre-trained Transformer 2
- **BLIP-2**: Salesforce's Bootstrapping Language-Image Pre-training
- **Hugging Face Transformers**: For the excellent transformer library
- **COCO Dataset**: Microsoft Common Objects in Context
- **NoCaps Dataset**: For zero-shot evaluation benchmark

---

**Last Updated**: April 2026






