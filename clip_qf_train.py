import argparse
import os
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from transformers import Blip2Model
import torch
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader, random_split
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
import sys
from typing import Tuple, Optional, Union
from cliptrain import TransformerMapper, ClipCocoDataset
from cliptrain import MLP
from peft import LoraConfig, get_peft_model
# from predict_ora import generate2, generate_beam
import torch.optim as optim
from torch.optim import AdamW

def set_seed(seed=42):
    """
    设置所有随机种子以确保可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


def get_device():
    """
    自动选择可用设备：优先 cuda:0，其次 cuda:1，最后 cpu
    """
    devices_to_try = ['cuda:0', 'cuda:1', 'cpu']

    for device_name in devices_to_try:
        try:
            if device_name == 'cpu':
                return torch.device('cpu')

            # 检查 CUDA 是否可用
            if not torch.cuda.is_available():
                print(f"CUDA not available, skipping {device_name}")
                continue

            # 检查指定 GPU 是否存在
            gpu_id = int(device_name.split(':')[1])
            if gpu_id >= torch.cuda.device_count():
                print(f"GPU {gpu_id} not found (total: {torch.cuda.device_count()})")
                continue

            # 尝试分配小 tensor 测试 GPU 是否可用
            test_tensor = torch.zeros(1).to(device_name)
            del test_tensor
            torch.cuda.empty_cache()

            print(f"Using device: {device_name}")
            return torch.device(device_name)

        except RuntimeError as e:
            print(f"Failed to use {device_name}: {e}")
            continue

    # 所有尝试都失败，返回 cpu
    print("All GPUs unavailable, using CPU")
    return torch.device('cpu')


class StandaloneQFormer(nn.Module):
    """
    独立的 Q-Former 模块
    """

    def __init__(
            self,
            model_name="Salesforce/blip2-opt-2.7b",
            freeze_qformer=None,
            output_dim=None  # 自定义输出维度
    ):
        super().__init__()

        # 加载预训练BLIP-2
        print("Loading BLIP-2 model...")
        blip2 = Blip2Model.from_pretrained(model_name)

        # 提取组件
        self.qformer = blip2.qformer
        self.query_tokens = nn.Parameter(blip2.query_tokens.data.clone())
        self.ln_vision = nn.LayerNorm(blip2.config.vision_config.hidden_size)

        # 原始投影层
        self.original_proj = blip2.language_projection
        original_output_dim = self.original_proj.out_features

        # 自定义输出维度
        if output_dim and output_dim != original_output_dim:
            self.custom_proj = nn.Linear(768, output_dim)
            self.use_custom_proj = True
        else:
            self.custom_proj = None
            self.use_custom_proj = False

        # 冻结Q-Former
        if freeze_qformer:
            self._freeze_qformer()
        else:
            print("Q-Former parameters are trainable.")

        print(f"Q-Former loaded. Query tokens: {self.query_tokens.shape}")

    def _freeze_qformer(self):
        for param in self.qformer.parameters():
            param.requires_grad = False
        self.query_tokens.requires_grad = False
        print("Q-Former parameters frozen.")

    def forward(self, image_embeds, attention_mask=None):
        """
        Args:
            image_embeds: [B, N, D] 图像编码器输出
            attention_mask: [B, N] 注意力掩码
        Returns:
            output: [B, 32, output_dim]
        """
        batch_size = image_embeds.size(0)

        # LayerNorm
        image_embeds = self.ln_vision(image_embeds)

        # 创建注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(
                image_embeds.size()[:-1],
                dtype=torch.long,
                device=image_embeds.device
            )

        # 扩展 query tokens
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Q-Former 前向传播
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        query_output = query_outputs.last_hidden_state  # [B, 32, 768]

        # 投影
        if self.use_custom_proj:
            output = self.custom_proj(query_output)
        else:
            output = self.original_proj(query_output)

        return output


class AdaptiveQFormer(nn.Module):
    """
    适配不同图像编码器的 Q-Former
    """

    def __init__(
            self,
            image_encoder_dim,  # 你的图像编码器输出维度
            target_llm_dim,  # 目标LLM的输入维度
            model_name="Salesforce/blip2-opt-2.7b",
            freeze_qformer:bool = None
    ):
        super().__init__()

        # 原始Q-Former期望的输入维度 (ViT-G: 1408)
        self.qformer_input_dim = 1408

        # 输入适配层
        if image_encoder_dim != self.qformer_input_dim:
            self.input_adapter = nn.Sequential(
                nn.Linear(image_encoder_dim, self.qformer_input_dim),
                nn.LayerNorm(self.qformer_input_dim)
            )
        else:
            self.input_adapter = nn.Identity()

        # 加载 Q-Former
        self.qformer_module = StandaloneQFormer(
            model_name=model_name,
            freeze_qformer=freeze_qformer,
            output_dim=target_llm_dim
        )

    def forward(self, image_embeds, attention_mask=None):
        # 适配输入维度
        adapted_embeds = self.input_adapter(image_embeds)
        # 如果是全局特征 [B, D]，unsqueeze 为序列 [B, 1, D]
        if adapted_embeds.dim() == 2:
            adapted_embeds = adapted_embeds.unsqueeze(1)
        # Q-Former 处理
        return self.qformer_module(adapted_embeds, attention_mask)


# class ClipCaptionModelQFVE(nn.Module):
#
#     def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
#         if self.qformer_enabled:
#             # Q-Former 输出有 32 个查询令牌
#             return torch.zeros(batch_size, 32, dtype=torch.int64, device=device)
#         else:
#             return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
#
#     def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
#                 labels: Optional[torch.Tensor] = None):
#         # GPT-2 token embedding
#         embedding_text = self.gpt.transformer.wte(tokens)
#
#         if self.qformer_enabled:
#             # Q-Former 输出是 [B, 32, llm_dim]，需要适配到 GPT-2 维度
#             prefix_projections = self.clip_project(prefix)  # [B, 32, llm_dim]
#
#             # 将 Q-Former 输出投影到 GPT-2 嵌入维度
#             if prefix_projections.size(-1) != self.gpt_embedding_size:
#                 # 如果维度不匹配，添加一个投影层
#                 if not hasattr(self, 'qformer_proj'):
#                     self.qformer_proj = nn.Linear(prefix_projections.size(-1), self.gpt_embedding_size)
#                 prefix_projections = self.qformer_proj(prefix_projections)
#
#         else:
#             with torch.no_grad():
#                 prefix_projections = self.clip_project(prefix).view(
#                     -1, self.prefix_length, self.gpt_embedding_size
#                 )
#
#         # 拼接 prefix + text
#         embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
#
#         # labels 对齐 prefix
#         if labels is not None:
#             dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
#             # if self.qformer_enabled:
#             #     # Q-Former 模式下，prefix 长度是 32
#             #     dummy_token = torch.zeros(tokens.shape[0], prefix_projections.shape[1],
#             #                               dtype=torch.int64, device=tokens.device)
#             labels = torch.cat((dummy_token, tokens), dim=1)
#
#         # GPT-2 LoRA forward
#         out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
#         return out
#
#     def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
#                  num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, use_qformer: bool = True,
#                  llm_dim: int = 768, ):
#         super(ClipCaptionModelQFVE, self).__init__()
#         self.prefix_length = prefix_length
#
#         # -------------------------------------------------------
#         # ① GPT-2 → LoRA
#         # -------------------------------------------------------
#         base_model = GPT2LMHeadModel.from_pretrained("gpt2")
#         self.gpt_embedding_size = base_model.transformer.wte.weight.shape[1]
#
#         config = LoraConfig(
#             r=8,
#             lora_alpha=32,
#             target_modules=["q_proj", "c_proj"],  #["c_attn", "c_proj"],
#             lora_dropout=0.1,
#             bias="none",
#             task_type="CAUSAL_LM",
#         )
#         self.gpt = get_peft_model(base_model, config)
#
#         # -------------------------------------------------------
#         # ② mapping 网络（MLP 或 TransformerMapper）
#         #    → 构建选择是否冻结
#         # -------------------------------------------------------
#         if use_qformer:
#             # 使用 Q-Former 架构替代原来的映射网络
#             self.clip_project = AdaptiveQFormer(
#                 image_encoder_dim=prefix_size,  # CLIP 特征维度
#                 target_llm_dim=llm_dim,  # 目标 LLM 维度
#                 model_name="Salesforce/blip2-opt-2.7b",
#                 freeze_qformer=True
#             )
#             self.qformer_enabled = True
#         else:
#             self.qformer_enabled = False
#             if mapping_type == MappingType.MLP:
#                 self.clip_project = MLP(
#                     (prefix_size,
#                      (self.gpt_embedding_size * prefix_length) // 2,
#                      self.gpt_embedding_size * prefix_length)
#                 )
#             else:
#                 self.clip_project = TransformerMapper(
#                     prefix_size, self.gpt_embedding_size,
#                     prefix_length, clip_length, num_layers
#                 )
#
#         #  冻结 mapping（MLP or TransformerMapper）
#         # for p in self.clip_project.parameters():
#         #     p.requires_grad = False
#         #
#         # #  TransformerMapper 专用的 prefix_const 也冻结（如果存在）
#         # if hasattr(self.clip_project, "prefix_const"):
#         #     self.clip_project.prefix_const.requires_grad = False
#
#         # -------------------------------------------------------
#         # ③ 冻结 GPT-2 原生参数，仅训练 LoRA
#         # -------------------------------------------------------
#         for name, param in self.gpt.named_parameters():
#             if "lora" not in name:
#                 param.requires_grad = False
class ClipCaptionModelQFVE(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.qformer_enabled:
            # Q-Former 输出有 32 个查询令牌
            return torch.zeros(batch_size, 32, dtype=torch.int64, device=device)
        else:
            return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        # GPT-2 token embedding
        embedding_text = self.gpt.transformer.wte(tokens)

        if self.qformer_enabled:
            # Q-Former 输出是 [B, 32, llm_dim],需要适配到 GPT-2 维度
            prefix_projections = self.clip_project(prefix)  # [B, 32, llm_dim]

            # 将 Q-Former 输出投影到 GPT-2 嵌入维度
            if prefix_projections.size(-1) != self.gpt_embedding_size:
                # 如果维度不匹配，添加一个投影层
                if not hasattr(self, 'qformer_proj'):
                    self.qformer_proj = nn.Linear(prefix_projections.size(-1), self.gpt_embedding_size)
                prefix_projections = self.qformer_proj(prefix_projections)

        else:
            # with torch.no_grad():
            prefix_projections = self.clip_project(prefix).view(
                -1, self.prefix_length, self.gpt_embedding_size
            )

        # 拼接 prefix + text
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        # labels 对齐 prefix
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        # GPT-2 forward (支持 LoRA 或原始模型)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, freeze_qformer: bool, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.Transformer, use_qformer: bool = True,
                 llm_dim: int = 768, use_lora: bool = None, lora_r: int = 8, lora_alpha: int = 32,
                 lora_dropout: float = 0.1, lora_target_modules: list = None):
        super(ClipCaptionModelQFVE, self).__init__()
        self.prefix_length = prefix_length
        self.use_lora = use_lora

        # -------------------------------------------------------
        # ① GPT-2 模型加载 (支持 LoRA 或原始模型)
        # -------------------------------------------------------
        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = base_model.transformer.wte.weight.shape[1]

        if use_lora:
            # 配置 LoRA
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "c_proj"]

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.gpt = get_peft_model(base_model, config)
            print(f"LoRA enabled: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            print(f"Target modules: {lora_target_modules}")

            # 冻结 GPT-2 原生参数，仅训练 LoRA
            for name, param in self.gpt.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False
        else:
            # 不使用 LoRA，使用原始 GPT-2 模型
            self.gpt = base_model
            print("LoRA disabled, using original GPT-2 model")
            for param in self.gpt.parameters():
                param.requires_grad = False
            print("GPT-2 parameters frozen.")

        # -------------------------------------------------------
        # ② mapping 网络 (MLP 或 TransformerMapper)
        # -------------------------------------------------------
        if use_qformer:
            # 使用 Q-Former 架构替代原来的映射网络
            self.clip_project = AdaptiveQFormer(
                image_encoder_dim=prefix_size,  # CLIP 特征维度
                target_llm_dim=llm_dim,  # 目标 LLM 维度
                model_name="Salesforce/blip2-opt-2.7b",
                freeze_qformer=freeze_qformer
            )
            self.qformer_enabled = True
        else:
            self.qformer_enabled = False
            if mapping_type == MappingType.MLP:
                print("using Mlp")
                self.clip_project = MLP(
                    (prefix_size,
                     (self.gpt_embedding_size * prefix_length) // 2,
                     self.gpt_embedding_size * prefix_length)
                )
            else:
                print("using Transformer")
                self.clip_project = TransformerMapper(
                    prefix_size, self.gpt_embedding_size,
                    prefix_length, clip_length, num_layers
                )

        # 打印模型参数信息
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        clip_project_params = sum(p.numel() for p in self.clip_project.parameters())
        clip_project_trainable = sum(p.numel() for p in self.clip_project.parameters() if p.requires_grad)
        print(f"clip_project Total parameters: {clip_project_params:,}")
        print(f"clip_project Trainable parameters: {clip_project_trainable:,}")


def trainVE(dataset: ClipCocoDataset, model: ClipCaptionModelQFVE, args,
            lr: float = 1e-5, warmup_steps: int = 5000):
    output_dir = args.output_dir
    output_prefix = args.output_prefix
    set_seed(args.seed)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()
    print(device)
    log_file_path = os.path.join(output_dir, f"{output_prefix}_trainlog.txt")
    log_file = open(log_file_path, 'a', encoding='utf-8')

    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {total_trainable:,}")
    optimizer = optim.AdamW(trainable_params, lr=lr)
    # if args.use_lora:
    #     optimizer = optim.AdamW(
    #         [p for p in model.parameters() if p.requires_grad],
    #         lr=lr
    #     )
    # else:
    #     optimizer = optim.AdamW(
    #         model.clip_project.parameters(),
    #         lr=lr
    #     )
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    datasetval = ClipCocoDataset(args.valdata, prefix_length=32,
                                 normalize_prefix=args.normalize_prefix)
    val_dataloader = DataLoader(datasetval, batch_size=64, shuffle=False, drop_last=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()

        model.train()
        total_train_loss = 0.0
        train_samples = 0

        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)

            with torch.no_grad():
                prefix_proj_test = model.clip_project(prefix[:1])  # 测试单个样本
                actual_prefix_length = prefix_proj_test.shape[1]  # 应该是 32

            logits = outputs.logits[:, actual_prefix_length - 1: -1]
            # logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 累积训练损失
            batch_size_actual = tokens.size(0)
            total_train_loss += loss.item() * batch_size_actual
            train_samples += batch_size_actual

            progress.set_postfix({"loss": loss.item()})
            progress.update()
            # if (idx + 1) % 10000 == 0:
            #     torch.save(
            #         model.state_dict(),
            #         os.path.join(output_dir, f"{output_prefix}_latest.p t"),
            #     )
        progress.close()

        # 计算平均训练损失
        avg_train_loss = total_train_loss / train_samples
        print(f"Epoch {epoch} - Average training loss: {avg_train_loss}")
        print(f"Epoch {epoch} - Average training loss: {avg_train_loss}", file=log_file)
        log_file.flush()
        ##################################################################

        model.eval()

        total_val_loss = 0.0
        total_samples = 0
        with torch.no_grad():  # 禁用梯度计算
            val_progress = tqdm(total=len(val_dataloader), desc=f"Validation {output_prefix}")
            for idx, (tokens, mask, prefix) in enumerate(val_dataloader):
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix, mask)

                with torch.no_grad():
                    prefix_proj_test = model.clip_project(prefix[:1])
                    actual_prefix_length = prefix_proj_test.shape[1]

                logits = outputs.logits[:, actual_prefix_length - 1: -1]
                #logits = outputs.logits[:, dataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

                batch_size_actual = tokens.size(0)
                total_val_loss += loss.item() * batch_size_actual
                total_samples += batch_size_actual

                val_progress.set_postfix({"val_loss": loss.item()})
                val_progress.update()
            val_progress.close()

        avg_val_loss = total_val_loss / total_samples
        print(f"Epoch {epoch} - Average validation loss: {avg_val_loss}")
        print(f"Epoch {epoch} - Average validation loss: {avg_val_loss}", file=log_file)
        log_file.flush()
        #####################################################################
        # 只有当训练损失和验证损失都降低时才保存模型
        if avg_train_loss < best_train_loss and avg_val_loss < best_val_loss:
            best_train_loss = avg_train_loss
            best_val_loss = avg_val_loss
            print(f"Saving model - improved train loss: {avg_train_loss}, val loss: {avg_val_loss}")
            print(f"Saving model - improved train loss: {avg_train_loss}, val loss: {avg_val_loss}", file=log_file)
            log_file.flush()
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}_best.pt"),
            )
        else:
            print(f"Not saving model - best train loss: {best_train_loss}, best val loss: {best_val_loss}")
            print(f"Not saving model - best train loss: {best_train_loss}, best val loss: {best_val_loss}",
                  file=log_file)
            log_file.flush()

        if epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    log_file.close()
    return model


def train_flickr(dataset: ClipCocoDataset, model: ClipCaptionModelQFVE, args,
            lr: float = 2e-5, warmup_steps: int = 5000):
    output_dir = args.output_dir
    output_prefix = args.output_prefix
    set_seed(args.seed)
    device = get_device()
    print(device)
    log_file_path = os.path.join(output_dir, f"{output_prefix}_trainlog.txt")
    log_file = open(log_file_path, 'a', encoding='utf-8')

    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {total_trainable:,}")
    optimizer = optim.AdamW(trainable_params, lr=lr)

    from torch.utils.data import Subset
    subset_indices = list(range(min(150000, len(dataset))))
    subset_dataset = Subset(dataset, subset_indices)
    print(f"Using {len(subset_dataset)} samples for training (first 30000 images)")

    train_dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    best_train_loss = float('inf')
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()

        model.train()
        total_train_loss = 0.0
        train_samples = 0

        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)

            with torch.no_grad():
                prefix_proj_test = model.clip_project(prefix[:1])
                actual_prefix_length = prefix_proj_test.shape[1]

            logits = outputs.logits[:, actual_prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            batch_size_actual = tokens.size(0)
            total_train_loss += loss.item() * batch_size_actual
            train_samples += batch_size_actual

            progress.set_postfix({"loss": loss.item()})
            progress.update()
        progress.close()

        avg_train_loss = total_train_loss / train_samples
        print(f"Epoch {epoch} - Average training loss: {avg_train_loss}")
        print(f"Epoch {epoch} - Average training loss: {avg_train_loss}", file=log_file)
        log_file.flush()

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            print(f"Saving model - improved train loss: {avg_train_loss}")
            print(f"Saving model - improved train loss: {avg_train_loss}", file=log_file)
            log_file.flush()
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}_best.pt"),
            )
        else:
            print(f"Not saving model - best train loss: {best_train_loss}")
            print(f"Not saving model - best train loss: {best_train_loss}", file=log_file)
            log_file.flush()

        if epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    log_file.close()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/oscar_split_ViT-B_32_train.pkl')
    parser.add_argument('--valdata', default='', required=True)
    parser.add_argument('--output_dir', default='./checkpoints/clip_qf_ve/')
    parser.add_argument('--output_prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=32)
    parser.add_argument('--clip_length', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--re_train', dest='re_train', action='store_true')
    parser.add_argument('--use_qformer', action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--freeze_qformer', action='store_true', help='If set, freeze Q-Former parameters')
    parser.add_argument('--reloadmodel', type=str, default=None)
    parser.add_argument('--train_flickr', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.use_qformer:
        use_qformer = True
    else:
        use_qformer = False
    if args.use_lora:
        use_lora = True
    else:
        use_lora = False

    model = ClipCaptionModelQFVE(
        freeze_qformer=args.freeze_qformer,
        prefix_length=prefix_length,
        clip_length=args.clip_length,
        prefix_size=prefix_dim,
        use_qformer=use_qformer,
        llm_dim=768,
        mapping_type=MappingType.Transformer,
        use_lora=use_lora,
        lora_r=8,  # LoRA 秩
        lora_alpha=32,  # LoRA alpha 参数
        lora_dropout=0.1
    )
    if args.re_train:
        model.load_state_dict(torch.load(args.reloadmodel, map_location='cpu'),
                              strict=False)
    if args.train_flickr:
        train_flickr(dataset, model, args)
    else:
        trainVE(dataset, model, args)


if __name__ == "__main__":
    main()
