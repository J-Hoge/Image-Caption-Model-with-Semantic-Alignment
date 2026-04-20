import argparse
import os
from PIL import Image
from clip import clip
from torch import nn
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import skimage.io as io
from torch.nn import functional as nnf
import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
import pickle
import sys
from typing import Tuple, Optional, Union
from cliptrain import TransformerMapper, ClipCocoDataset
from cliptrain import MLP
from peft import LoraConfig, get_peft_model
from transformers import GPT2LMHeadModel
from predict_ora import generate2, generate_beam
import torch.optim as optim


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCaptionModelVE(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        # GPT-2 token embedding
        embedding_text = self.gpt.transformer.wte(tokens)

        # Freeze mapping: clip_project is not trained and only used for forward pass
        with torch.no_grad():
            prefix_projections = self.clip_project(prefix).view(
                -1, self.prefix_length, self.gpt_embedding_size
            )

        # 拼接 prefix + text
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        # labels 对齐 prefix
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        # GPT-2 LoRA forward
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModelVE, self).__init__()
        self.prefix_length = prefix_length

        # -------------------------------------------------------
        # ① GPT-2 → LoRA
        # -------------------------------------------------------
        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = base_model.transformer.wte.weight.shape[1]

        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "c_proj"],  #["c_attn", "c_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.gpt = get_peft_model(base_model, config)

        # -------------------------------------------------------
        # ② mapping 网络（MLP 或 TransformerMapper）
        #    → 构建选择是否冻结
        # -------------------------------------------------------
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP(
                (prefix_size,
                 (self.gpt_embedding_size * prefix_length) // 2,
                 self.gpt_embedding_size * prefix_length)
            )
        else:
            self.clip_project = TransformerMapper(
                prefix_size, self.gpt_embedding_size,
                prefix_length, clip_length, num_layers
            )

        #  冻结 mapping（MLP or TransformerMapper）
        # for p in self.clip_project.parameters():
        #     p.requires_grad = False
        #
        # #  TransformerMapper 专用的 prefix_const 也冻结（如果存在）
        # if hasattr(self.clip_project, "prefix_const"):
        #     self.clip_project.prefix_const.requires_grad = False

        # -------------------------------------------------------
        # ③ 冻结 GPT-2 原生参数，仅训练 LoRA
        # -------------------------------------------------------
        for name, param in self.gpt.named_parameters():
            if "lora" not in name:
                param.requires_grad = False


def trainVE(dataset: ClipCocoDataset, model: ClipCaptionModelVE, args,
            lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):
    output_prefix = "coco_prefix_trainandval_VE"
    # output_dir = "./checkpoints"
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    datasetval = ClipCocoDataset('./data/oscar_split_RN50x4_val.pkl', prefix_length=40,
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
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
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
        ##################################################################

        model.eval()

        total_val_loss = 0.0
        total_samples = 0
        with torch.no_grad():  # 禁用梯度计算
            val_progress = tqdm(total=len(val_dataloader), desc=f"Validation {output_prefix}")
            for idx, (tokens, mask, prefix) in enumerate(val_dataloader):
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, dataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

                batch_size_actual = tokens.size(0)
                total_val_loss += loss.item() * batch_size_actual
                total_samples += batch_size_actual

                val_progress.set_postfix({"val_loss": loss.item()})
                val_progress.update()
            val_progress.close()

        avg_val_loss = total_val_loss / total_samples
        print(f"Epoch {epoch} - Average validation loss: {avg_val_loss}")
        #####################################################################
        # 只有当训练损失和验证损失都降低时才保存模型
        if avg_train_loss < best_train_loss and avg_val_loss < best_val_loss:
            best_train_loss = avg_train_loss
            best_val_loss = avg_val_loss
            print(f"Saving model - improved train loss: {avg_train_loss}, val loss: {avg_val_loss}")
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}_best.pt"),
            )
        else:
            print(f"Not saving model - best train loss: {best_train_loss}, best val loss: {best_val_loss}")

        if epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model


def reloadoradata_test(use_beam_search):
    model_path = ("checkpoints/transformer_weights.pt")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prefix_length = 40

    model = ClipCaptionModelVE(prefix_length, clip_length=40, prefix_size=640, num_layers=8, mapping_type='transformer')
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model = model.eval()
    model = model.to(device)

    image = io.imread('Images/COCO_val2014_000000354533.jpg')
    image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix = prefix / prefix.norm(2, -1).item()
        print(prefix.size())
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    if use_beam_search:
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)
    else:
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    print('\n')
    print(generated_text_prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/oscar_split_ViT-B_32_train.pkl')
    parser.add_argument('--out_dir', default='./checkpoints/clip_ve/')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--re_trian', dest='re_train', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    model = ClipCaptionModelVE(prefix_length, clip_length=40, prefix_size=640, num_layers=8, mapping_type='transformer')
    if args.re_train:
        model.load_state_dict(torch.load(args.out_dir + '/' + 'coco_prefix_trainandval_VE_best.pt', map_location='cpu'), strict=False)
    trainVE(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == "__main__":
    main()
