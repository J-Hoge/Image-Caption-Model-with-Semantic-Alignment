from datasets import load_dataset
from clip_qf_train import trainVE, ClipCaptionModelQFVE, MappingType
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import pickle
import os
import numpy as np
from tqdm import tqdm
import clip
from PIL import Image
import requests
from io import BytesIO


class ConceptualCaptionsDataset(Dataset):
    """
    Conceptual Captions 数据集加载器
    图像只在内存中处理，不保存到本地
    """

    def __len__(self) -> int:
        return len(self.clip_embeddings)

    def __getitem__(self, item: int):
        tokens = self.captions_tokens[item]
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)

        prefix = self.clip_embeddings[item]
        if self.normalize_prefix:
            prefix = prefix / prefix.norm(2, -1)

        return tokens, mask, prefix

    def __init__(self, data_path=None, hf_dataset=None, prefix_length=32,
                 gpt2_type="gpt2", normalize_prefix=False, max_samples=None,
                 device='cuda'):
        """
        Args:
            data_path: 预处理的 pkl 文件路径（如果已预处理）
            hf_dataset: HuggingFace dataset 对象（如果直接加载）
            prefix_length: prefix 长度
            gpt2_type: GPT-2 模型类型
            normalize_prefix: 是否归一化 CLIP 特征
            max_samples: 最大样本数（用于测试）
            device: 设备
        """
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.device = device

        # 加载 CLIP 模型用于提取图像特征
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=device)
        self.clip_model.eval()

        if data_path and os.path.exists(data_path):
            # 从预处理的 pkl 文件加载
            print(f"Loading preprocessed data from {data_path}")
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.clip_embeddings = data['clip_embeddings']
            self.captions_tokens = data['captions_tokens']

        elif hf_dataset is not None:
            # 从 HuggingFace dataset 实时处理，只保存在内存中
            print("Processing HuggingFace dataset (images will NOT be saved to disk)...")
            self._process_hf_dataset(hf_dataset, max_samples)
        else:
            raise ValueError("Either data_path or hf_dataset must be provided")

        print(f"Dataset size: {len(self)}")

    def _process_hf_dataset(self, dataset, max_samples=None):
        """
        处理 HuggingFace dataset，提取 CLIP 特征和 tokenization
        图像只在内存中处理，不保存到本地
        """
        self.clip_embeddings = []
        self.captions_tokens = []

        # 限制样本数量（用于测试）
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        print(f"Processing {len(dataset)} samples...")
        print("Note: Images are processed in memory only and will NOT be saved to disk.")

        for idx in tqdm(range(len(dataset))):
            try:
                item = dataset[idx]
                image_url = item['image_url']
                caption = item['caption']

                # 下载并处理图像（仅在内存中）
                image = self._load_image_from_url(image_url)
                if image is None:
                    continue

                # 提取 CLIP 特征
                with torch.no_grad():
                    image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                    image_features = self.clip_model.encode_image(image_input)
                    image_features = image_features.cpu().numpy().flatten()

                # Tokenize caption
                tokens = torch.tensor(
                    self.tokenizer.encode(caption),
                    dtype=torch.int64
                )

                self.clip_embeddings.append(torch.tensor(image_features, dtype=torch.float32))
                self.captions_tokens.append(tokens)

                # 立即删除图像对象，释放内存
                del image

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

            # 每处理 1000 个样本打印一次进度
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} samples, successful: {len(self.clip_embeddings)}")

        if len(self.clip_embeddings) == 0:
            raise ValueError("No valid samples processed!")

        # 转换为 tensor
        self.clip_embeddings = torch.stack(self.clip_embeddings)

        # # 可选：保存预处理的数据（只保存 CLIP 特征和 tokens，不保存原始图像）
        # save_choice = input("\nSave preprocessed features to pkl file? (y/n, default: n): ").strip().lower()
        # if save_choice == 'y':
        #     save_path = f"./data/conceptual_captions_processed_{len(self.clip_embeddings)}.pkl"
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     with open(save_path, 'wb') as f:
        #         pickle.dump({
        #             'clip_embeddings': self.clip_embeddings,
        #             'captions_tokens': self.captions_tokens
        #         }, f)
        #     print(f"Preprocessed data saved to {save_path}")
        #     print("This file only contains CLIP features and tokens, NOT original images.")
        # else:
        #     print("Features will NOT be saved. They will be lost after training.")

    def _load_image_from_url(self, url, timeout=10):
        """
        从 URL 加载图像到内存（不保存到本地）
        """
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content)).convert('RGB')
                return image
        except Exception as e:
            print(f"Failed to load image from {url}: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_preprocessed', action='store_true',
                       help='Use preprocessed pkl file instead of downloading')
    parser.add_argument('--preprocessed_path', type=str,
                       default='./data/conceptual_captions_processed.pkl',
                       help='Path to preprocessed pkl file')
    parser.add_argument('--output_dir', type=str,
                       default='./checkpoints/conceptual/')
    parser.add_argument('--output_prefix', type=str,
                       default='conceptual_prefix')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--prefix_length', type=int, default=32)
    parser.add_argument('--clip_length', type=int, default=40)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--normalize_prefix', action='store_true')
    parser.add_argument('--use_qformer', action='store_true',
                       help='Use Q-Former architecture')
    parser.add_argument('--use_lora', action='store_true',
                       help='Use LoRA for GPT-2')
    parser.add_argument('--freeze_qformer', action='store_true',
                       help='Freeze Q-Former parameters')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to process (for testing)')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                       help='Validation set ratio')
    parser.add_argument('--save_features', action='store_true',
                       help='Save CLIP features to pkl file after preprocessing')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to pretrained model checkpoint for fine-tuning')
    parser.add_argument('--strict_load', action='store_true',
                       help='Use strict mode when loading checkpoint')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据集
    if args.use_preprocessed:
        # 使用预处理的数据
        print("Loading preprocessed dataset...")
        full_dataset = ConceptualCaptionsDataset(
            data_path=args.preprocessed_path,
            prefix_length=args.prefix_length,
            normalize_prefix=args.normalize_prefix,
            device=device
        )
    else:
        # 从 HuggingFace 加载原始数据
        print("Loading Conceptual Captions from HuggingFace...")
        ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")

        # 分割训练集和验证集
        train_data = ds['train']

        # 创建训练集
        print("Creating training dataset...")
        print("Images will be downloaded to memory, processed, and then discarded.")
        full_dataset = ConceptualCaptionsDataset(
            hf_dataset=train_data,
            prefix_length=args.prefix_length,
            normalize_prefix=args.normalize_prefix,
            max_samples=args.max_samples,
            device=device
        )

    # 分割训练集和验证集
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 如果需要，保存验证集到单独的文件
    if args.save_features:
        val_data_path = args.preprocessed_path.replace('.pkl', '_val.pkl')
        os.makedirs(os.path.dirname(val_data_path), exist_ok=True)
        with open(val_data_path, 'wb') as f:
            pickle.dump({
                'clip_embeddings': val_dataset.dataset.clip_embeddings[val_dataset.indices],
                'captions_tokens': [val_dataset.dataset.captions_tokens[i] for i in val_dataset.indices]
            }, f)
        print(f"Validation data saved to {val_data_path}")
    else:
        # 临时保存验证集用于训练
        val_data_path = './data/temp_conceptual_val.pkl'
        os.makedirs(os.path.dirname(val_data_path), exist_ok=True)
        with open(val_data_path, 'wb') as f:
            pickle.dump({
                'clip_embeddings': val_dataset.dataset.clip_embeddings[val_dataset.indices],
                'captions_tokens': [val_dataset.dataset.captions_tokens[i] for i in val_dataset.indices]
            }, f)
        print(f"Temporary validation data saved to {val_data_path}")

    # 创建模型
    prefix_dim = 512  # ViT-B/32 的输出维度

    mapping_type = MappingType.Transformer

    model = ClipCaptionModelQFVE(
        freeze_qformer=args.freeze_qformer,
        prefix_length=args.prefix_length,
        clip_length=args.clip_length,
        prefix_size=prefix_dim,
        use_qformer=args.use_qformer,
        llm_dim=768,
        mapping_type=mapping_type,
        use_lora=args.use_lora,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # 加载预训练模型参数（如果提供）
    if args.resume_from:
        print(f"\n{'='*60}")
        print(f"Loading pretrained model from: {args.resume_from}")
        print(f"{'='*60}")

        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_from}")

        # 加载 checkpoint
        checkpoint = torch.load(args.resume_from, map_location=device)

        # 处理不同的 checkpoint 格式
        if isinstance(checkpoint, dict):
            # 如果是 state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Found 'model_state_dict' in checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Found 'state_dict' in checkpoint")
            else:
                state_dict = checkpoint
                print("Loading raw state_dict")
        else:
            state_dict = checkpoint

        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=args.strict_load)

        if missing_keys:
            print(f"\nMissing keys ({len(missing_keys)}):")
            for key in missing_keys[:10]:  # 只显示前10个
                print(f"  - {key}")
            if len(missing_keys) > 10:
                print(f"  ... and {len(missing_keys) - 10} more")

        if unexpected_keys:
            print(f"\nUnexpected keys ({len(unexpected_keys)}):")
            for key in unexpected_keys[:10]:  # 只显示前10个
                print(f"  - {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... and {len(unexpected_keys) - 10} more")

        if not missing_keys and not unexpected_keys:
            print("\n✓ All weights loaded successfully!")
        else:
            print(f"\n⚠ Model loaded with some mismatches (strict={args.strict_load})")

        # 打印可训练参数信息
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100*trainable_params/total_params:.2f}%)")
        print(f"{'='*60}\n")

    # 训练配置
    class TrainArgs:
        def __init__(self, args, val_path):
            self.output_dir = args.output_dir
            self.output_prefix = args.output_prefix
            self.bs = args.bs
            self.epochs = args.epochs
            self.valdata = val_path
            self.normalize_prefix = args.normalize_prefix

    train_args = TrainArgs(args, val_data_path)

    # 开始训练
    print("Starting training...")
    if args.resume_from:
        print(f"Fine-tuning from: {args.resume_from}")
    else:
        print("Training from scratch")

    trainVE(train_dataset, model, train_args)

    # 训练结束后，如果不保存特征，删除临时文件
    if not args.save_features and not args.use_preprocessed:
        if os.path.exists(val_data_path):
            os.remove(val_data_path)
            print(f"Temporary validation file removed: {val_data_path}")


if __name__ == "__main__":
    main()

