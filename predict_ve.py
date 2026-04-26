from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from datasets import load_dataset
from PIL import Image
import clip
import skimage.io as io
import numpy as np
from io import BytesIO
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple
from transformers import GPT2Tokenizer
from tqdm import tqdm, trange
from enum import Enum
from cliptrain import ClipCaptionPrefix, Dataset, ClipCocoDataset
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
import argparse
# from lora_for_VE import ClipCaptionModelVE
from clip_qf_train import ClipCaptionModelQFVE
from cliptrain import ClipCaptionPrefix

CPU = torch.device('cpu')


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset_test(Dataset):

    def __len__(self) -> int:
        return len(self.prefixes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        prefix = self.prefixes[idx]
        return prefix, self.image_ids[idx]

    def __init__(self, data_path: str, prefix_length: int, normalize_prefix=False):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            # {
            #     "clip_embedding": tensor of shape[N, 512],  # N个图像的CLIP特征拼接
            #     "images": [
            #         {"image_id": 123, "clip_embedding": 0},
            #         {"image_id": 456, "clip_embedding": 1},
            #         ...
            #     ]
            # }
        print("Data size is %0d" % len(data["clip_embedding"]))
        self.prefixes = data["clip_embedding"]  # 图像特征
        self.image_ids = [img["image_id"] for img in data["images"]]  # 图像ID列表

        if normalize_prefix:
            self.prefixes = self.prefixes / self.prefixes.norm(2, -1).unsqueeze(-1)

        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        sys.stdout.flush()


# def generate_beam_for_cider(
#         model, tokenizer, embed=None,
#         beam_size=3,
#         max_length=20,  # ← 关键：根据数据集平均长度设置
#         min_length=8,  # ← 强制最小长度
#         temperature=1.0,  # ← CIDEr 优化通常用 1.0
#         length_penalty=0.0,  # ← 重要：CIDEr 已考虑长度，不需额外惩罚
#         repetition_penalty=1.2,  # ← 降低到 1.2，允许合理重复
#         no_repeat_ngram_size=2,  # ← 关闭 n-gram 禁止（可能过于严格）
#         diversity_penalty=0.0,  # ← 新增：beam diversity
#         use_nucleus_sampling=False,  # ← 可选：top-p 采样
#         ban_special_chars=True,  # ← 可选开关
# ):
#     model.eval()
#     device = next(model.parameters()).device
#
#     # ========== 可选：禁止特殊字符 ==========
#     bad_tokens = set()
#     if ban_special_chars:
#         for char in ['\n', '\r', '\t', '\xa0']:
#             token_ids = tokenizer.encode(char, add_special_tokens=False)
#             bad_tokens.update(token_ids)
#
#     # ========== 辅助函数 ==========
#     def apply_length_constraint(logits, current_length, min_len, max_len):
#         """长度约束：未达到 min_len 时禁止 EOS"""
#         if current_length < min_len:
#             eos_id = tokenizer.eos_token_id
#             if eos_id is not None:
#                 logits[:, eos_id] = float('-inf')
#             # 也禁止句号等结束符
#             period_ids = tokenizer.encode('.', add_special_tokens=False)
#             for pid in period_ids:
#                 logits[:, pid] = float('-inf')
#         return logits
#
#     def apply_repetition_penalty(logits, token_ids, penalty):
#         """温和的重复惩罚"""
#         if penalty == 1.0 or len(token_ids) == 0:
#             return logits
#
#         for i, tokens in enumerate(token_ids):
#             if len(tokens) == 0:
#                 continue
#             for token_id in set(tokens):
#                 # 只惩罚连续重复 >= 2 次的情况
#                 if len(tokens) >= 2 and tokens[-1] == token_id and tokens[-2] == token_id:
#                     if logits[i, token_id] > 0:
#                         logits[i, token_id] /= penalty
#                     else:
#                         logits[i, token_id] *= penalty
#         return logits
#
#     def apply_diversity_penalty(logits, beams, diversity_penalty):
#         """Diverse Beam Search: 惩罚当前 step 已选择的 token"""
#         if diversity_penalty == 0.0:
#             return logits
#
#         # 收集当前 step 已选的 token（来自其他 beam）
#         already_selected = set()
#         for tokens, _, _ in beams:
#             if len(tokens) > 0:
#                 already_selected.add(tokens[-1])
#
#         for token_id in already_selected:
#             logits[:, token_id] -= diversity_penalty
#
#         return logits
#
#     # ========== Beam Search 主循环 ==========
#     with torch.no_grad():
#         # Prefix embedding
#         if embed is not None:
#             prefix_embed = embed
#         else:
#             prefix_embed = None
#
#         # 初始化 beams
#         beams = [([], 0.0, False)]  # (tokens, score, is_finished)
#         finished_beams = []
#
#         for step in range(max_length):
#             all_candidates = []
#
#             for beam_idx, (token_ids, score, is_finished) in enumerate(beams):
#                 if is_finished:
#                     all_candidates.append((token_ids, score, True))
#                     continue
#
#                 # 构建输入
#                 if len(token_ids) > 0:
#                     tokens_tensor = torch.tensor([token_ids], device=device)
#                     token_embed = model.gpt.transformer.wte(tokens_tensor)
#                 else:
#                     token_embed = torch.empty(1, 0, model.gpt.config.n_embd, device=device)
#
#                 if prefix_embed is not None:
#                     inputs_embeds = torch.cat([prefix_embed, token_embed], dim=1)
#                 else:
#                     inputs_embeds = token_embed
#
#                 # 前向
#                 outputs = model.gpt(inputs_embeds=inputs_embeds)
#                 logits = outputs.logits[:, -1, :].clone()
#
#                 # ========== 关键修改：更温和的策略 ==========
#
#                 # 1. 长度约束（确保生成足够长）
#                 logits = apply_length_constraint(logits, len(token_ids), min_length, max_length)
#
#                 # 2. 温和的重复惩罚（只惩罚连续重复）
#                 logits = apply_repetition_penalty(logits, [token_ids], repetition_penalty)
#
#                 # 3. 可选：禁止特殊字符
#                 if ban_special_chars:
#                     for tid in bad_tokens:
#                         if tid < logits.shape[1]:
#                             logits[0, tid] = float('-inf')
#
#                 # 4. Diversity penalty（可选）
#                 if diversity_penalty > 0 and beam_idx > 0:
#                     logits = apply_diversity_penalty(logits, beams[:beam_idx], diversity_penalty)
#
#                 # 5. 温度
#                 logits = logits / temperature
#
#                 # 6. 计算概率
#                 if use_nucleus_sampling:
#                     # Top-p sampling
#                     sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
#                     cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
#                     sorted_indices_to_remove = cumulative_probs > 0.9
#                     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#                     sorted_indices_to_remove[..., 0] = 0
#                     indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
#                     logits[indices_to_remove] = float('-inf')
#
#                 log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
#
#                 # 取 top-k
#                 top_log_probs, top_indices = log_probs.topk(beam_size * 2, dim=-1)
#
#                 for i in range(top_log_probs.shape[1]):
#                     token_id = top_indices[0, i].item()
#                     log_prob = top_log_probs[0, i].item()
#
#                     if log_prob == float('-inf'):
#                         continue
#
#                     new_tokens = token_ids + [token_id]
#                     new_score = score + log_prob
#
#                     # 检查是否结束
#                     is_eos = (token_id == tokenizer.eos_token_id)
#                     is_period = (token_id in tokenizer.encode('.', add_special_tokens=False))
#                     is_end = (is_eos or is_period) and len(new_tokens) >= min_length
#
#                     if is_end:
#                         finished_beams.append((new_tokens, new_score, True))
#                     else:
#                         all_candidates.append((new_tokens, new_score, False))
#
#             if not all_candidates:
#                 break
#
#             # ========== 关键：CIDEr 通常不需要长度归一化 ==========
#             # 因为 CIDEr 内部已考虑长度
#             if length_penalty != 0.0:
#                 def score_fn(candidate):
#                     tokens, score, _ = candidate
#                     length = max(len(tokens), 1)
#                     lp = ((5 + length) / 6) ** length_penalty
#                     return score / lp
#             else:
#                 def score_fn(candidate):
#                     return candidate[1]  # 直接用累积分数
#
#             all_candidates.sort(key=score_fn, reverse=True)
#             beams = all_candidates[:beam_size]
#
#         # 收集所有结果
#         for tokens, score, _ in beams:
#             if len(tokens) >= min_length:
#                 finished_beams.append((tokens, score, True))
#
#         # 最终排序
#         if length_penalty != 0.0:
#             finished_beams.sort(
#                 key=lambda x: x[1] / (((5 + len(x[0])) / 6) ** length_penalty),
#                 reverse=True
#             )
#         else:
#             finished_beams.sort(key=lambda x: x[1], reverse=True)
#
#     # 解码
#     output_texts = []
#     for tokens, score, _ in finished_beams[:beam_size]:
#         if len(tokens) > 0:
#             text = tokenizer.decode(tokens, skip_special_tokens=True)
#             # 最小清理
#             text = text.replace('\xa0', ' ').strip()
#             output_texts.append(text)
#
#     return output_texts[0] if output_texts else [""]
def generate_beam_for_cider(
        model, tokenizer, embed=None,
        beam_size=5,
        max_length=20,
        min_length=5,
        temperature=1.0,
        length_penalty=0.0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        diversity_penalty=0.0,
        use_nucleus_sampling=False,
        ban_special_chars=True,
):
    model.eval()
    device = next(model.parameters()).device

    bad_tokens = set()
    if ban_special_chars:
        for char in ['\n', '\r', '\t', '\xa0']:
            token_ids = tokenizer.encode(char, add_special_tokens=False)
            bad_tokens.update(token_ids)

    def apply_length_constraint(logits, current_length, min_len, max_len):
        """长度约束：未达到 min_len 时禁止 EOS"""
        if current_length < min_len:
            eos_id = tokenizer.eos_token_id
            if eos_id is not None:
                logits[:, eos_id] = float('-inf')
        return logits

    def apply_repetition_penalty(logits, token_ids, penalty):
        """改进的重复惩罚：惩罚所有历史重复token"""
        if penalty == 1.0 or len(token_ids) == 0:
            return logits

        for i, tokens in enumerate(token_ids):
            if len(tokens) == 0:
                continue
            from collections import Counter
            token_counts = Counter(tokens)
            for token_id, count in token_counts.items():
                if count >= 2:
                    if logits[i, token_id] > 0:
                        logits[i, token_id] /= (penalty ** count)
                    else:
                        logits[i, token_id] *= (penalty ** count)
        return logits

    def apply_no_repeat_ngram(logits, token_ids, ngram_size):
        """禁止生成会导致n-gram重复的token"""
        if ngram_size <= 1 or len(token_ids) < ngram_size - 1:
            return logits

        for i, tokens in enumerate(token_ids):
            if len(tokens) < ngram_size - 1:
                continue

            # 获取当前末尾的 (ngram_size-1) 个token作为前缀
            prefix_tuple = tuple(tokens[-(ngram_size - 1):])

            # 查找历史中所有该前缀后跟随的token
            banned_tokens = set()
            for j in range(len(tokens) - ngram_size + 1):
                if tuple(tokens[j:j + ngram_size - 1]) == prefix_tuple:
                    banned_tokens.add(tokens[j + ngram_size - 1])

            # 将这些token的概率设为-inf
            for token_id in banned_tokens:
                if token_id < logits.shape[1]:
                    logits[i, token_id] = float('-inf')

        return logits

    def apply_diversity_penalty(logits, beams, diversity_penalty):
        """Diverse Beam Search: 惩罚当前 step 已选择的 token"""
        if diversity_penalty == 0.0 or len(beams) == 0:
            return logits

        already_selected = set()
        for tokens, _, _ in beams:
            if len(tokens) > 0:
                already_selected.add(tokens[-1])

        for token_id in already_selected:
            if token_id < logits.shape[1]:
                logits[:, token_id] -= diversity_penalty

        return logits

    with torch.no_grad():
        if embed is not None:
            prefix_embed = embed
        else:
            prefix_embed = None

        beams = [([], 0.0, False)]
        finished_beams = []

        for step in range(max_length):
            all_candidates = []

            if all(is_finished for _, _, is_finished in beams) and finished_beams:
                break

            for beam_idx, (token_ids, score, is_finished) in enumerate(beams):
                if is_finished:
                    all_candidates.append((token_ids, score, True))
                    continue

                if len(token_ids) > 0:
                    tokens_tensor = torch.tensor([token_ids], device=device)
                    token_embed = model.gpt.transformer.wte(tokens_tensor)
                else:
                    token_embed = torch.empty(1, 0, model.gpt.config.n_embd, device=device)

                if prefix_embed is not None:
                    inputs_embeds = torch.cat([prefix_embed, token_embed], dim=1)
                else:
                    inputs_embeds = token_embed

                outputs = model.gpt(inputs_embeds=inputs_embeds)
                logits = outputs.logits[:, -1, :].clone()

                # 1. 长度约束
                logits = apply_length_constraint(logits, len(token_ids), min_length, max_length)

                # 2. 重复惩罚
                logits = apply_repetition_penalty(logits, [token_ids], repetition_penalty)

                # 3. N-gram重复禁止（新增）
                if no_repeat_ngram_size > 1:
                    logits = apply_no_repeat_ngram(logits, [token_ids], no_repeat_ngram_size)

                # 4. 禁止特殊字符
                if ban_special_chars:
                    for tid in bad_tokens:
                        if tid < logits.shape[1]:
                            logits[0, tid] = float('-inf')

                # 5. Diversity penalty
                if diversity_penalty > 0:
                    logits = apply_diversity_penalty(logits, beams, diversity_penalty)

                # 6. 温度缩放
                logits = logits / temperature

                # 7. Top-p 采样
                if use_nucleus_sampling:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > 0.9
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                num_valid_candidates = (log_probs > float('-inf')).sum().item()
                expand_size = min(beam_size * 2, num_valid_candidates)

                if expand_size <= 0:
                    continue

                top_log_probs, top_indices = log_probs.topk(expand_size, dim=-1)

                for i in range(top_log_probs.shape[1]):
                    token_id = top_indices[0, i].item()
                    log_prob = top_log_probs[0, i].item()

                    if log_prob == float('-inf'):
                        continue

                    new_tokens = token_ids + [token_id]
                    new_score = score + log_prob

                    is_eos = (token_id == tokenizer.eos_token_id)
                    is_period = (token_id in tokenizer.encode('.', add_special_tokens=False))
                    is_end = (is_eos or is_period) and len(new_tokens) >= min_length

                    if is_end:
                        finished_beams.append((new_tokens, new_score, True))
                    else:
                        all_candidates.append((new_tokens, new_score, False))

            if not all_candidates:
                break

            if length_penalty != 0.0:
                def score_fn(candidate):
                    tokens, score, _ = candidate
                    length = max(len(tokens), 1)
                    lp = ((5 + length) / 6) ** length_penalty
                    return score / lp
            else:
                def score_fn(candidate):
                    return candidate[1]

            all_candidates.sort(key=score_fn, reverse=True)
            beams = all_candidates[:beam_size]

        for tokens, score, is_finished in beams:
            if not is_finished and len(tokens) >= min_length:
                finished_beams.append((tokens, score, True))

        if length_penalty != 0.0:
            finished_beams.sort(
                key=lambda x: x[1] / (((5 + len(x[0])) / 6) ** length_penalty),
                reverse=True
            )
        else:
            finished_beams.sort(key=lambda x: x[1], reverse=True)

    output_texts = []
    for tokens, score, _ in finished_beams[:beam_size]:
        if len(tokens) > 0:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            text = text.replace('\xa0', ' ').strip()
            output_texts.append(text)

    return output_texts[0] if output_texts else ""


def generate_beam(
        model,
        tokenizer,
        beam_size: int = 5,
        prompt=None,
        embed=None,
        entry_length=67,
        temperature=1.0,
        stop_token: str = ".",
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(model, tokenizer, tokens=None, prompt=None, embed=None, entry_count=1, entry_length=67,
              # maximum number of words
              top_p=0.8, temperature=1., stop_token: str = '.',
              ):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                # next_token = torch.argmax(logits, -1).unsqueeze(0)
                probs = nnf.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def main(use_beam_search, model_path, model_type):
    # model_path = ("checkpoints/clip_ve/coco_prefix_trainandval_VE_best.pt")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # "ViT-B/16""RN50x4"
    clip_model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    #40
    # if model_type == 'clip_ve':
    #     prefix_length = 32
    #     model = ClipCaptionModelQFVE(prefix_length, clip_length=40, prefix_size=512, num_layers=8,use_qformer=False, use_lora=True,
    #                                mapping_type='transformer')
    #     model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    # elif model_type == 'clip_qf_ve':
    #     prefix_length = 32
    #     model = ClipCaptionModelQFVE(prefix_length, clip_length=40, prefix_size=512, num_layers=8,
    #                                  mapping_type='transformer', use_qformer=True, llm_dim=768)
    #     model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    if args.use_lora:
        use_lora = True
    else:
        use_lora = False
    if args.use_qformer:
        use_qformer = True
    else:
        use_qformer = False
    prefix_length = args.prefix_length
    model = ClipCaptionModelQFVE(prefix_length=args.prefix_length, clip_length=40, prefix_size=512, num_layers=8,
                                 mapping_type=MappingType.Transformer, freeze_qformer=True,
                                 use_qformer=use_qformer, use_lora=use_lora, llm_dim=768)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model = model.eval()
    model = model.to(device)

    image = io.imread('Images/COCO_val2014_000000060623.jpg')
    image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix = prefix / prefix.norm(2, -1).item()
        print(prefix.size())
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    if use_beam_search:
        generated_text_prefix = generate_beam_for_cider(
            model, tokenizer, embed=prefix_embed,
            beam_size=args.beam_size,
            max_length=20,
            min_length=8,
            temperature=1.0,
            length_penalty=0.0,
            repetition_penalty=1.2,  # 轻度惩罚
            ban_special_chars=True,
        )
    else:
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed, top_p=0.9, temperature=0.8)
    print('\n')
    print(generated_text_prefix)


def caption_folder(use_beam_search, model_path, image_folder, output_file):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 根据模型类型设置参数
    model = ClipCaptionPrefix(prefix_length=40, clip_length=40, prefix_size=640, num_layers=8,
                              mapping_type=MappingType.Transformer)

    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model = model.eval()
    model = model.to(device)

    # 获取文件夹中所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_folder) if f.lower().endswith(ext)])

    print(f"Found {len(image_files)} images in {image_folder}")

    # 生成描述
    generated_captions = []

    with torch.no_grad():
        progress = tqdm(total=len(image_files), desc="Generating captions")

        for idx, image_filename in enumerate(image_files):
            image_path = os.path.join(image_folder, image_filename)

            try:
                # 读取并预处理图片
                image = Image.open(image_path).convert('RGB')
                image = preprocess(image).unsqueeze(0).to(device)

                # 提取图像特征
                prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
                prefix = prefix / prefix.norm(2, -1).item()

                # 投影到模型输入维度
                prefix_embed = model.clip_project(prefix).reshape(1, 40, -1)

                # 生成文本
                if use_beam_search:
                    generated_text = generate_beam_for_cider(
                        model, tokenizer, embed=prefix_embed,
                        beam_size=5,
                        max_length=20,
                        min_length=8,
                        temperature=1.0,
                        length_penalty=0.0,
                        repetition_penalty=1.2,  # 轻度惩罚
                        ban_special_chars=True,
                    )
                else:
                    generated_text = generate2(model, tokenizer, embed=prefix_embed, top_p=0.9, temperature=0.8)

                # 保存结果（使用文件名作为 image_id）
                image_id = os.path.splitext(image_filename)[0]
                generated_captions.append({
                    "image_id": int(image_id),
                    "caption": generated_text.strip(),
                    "filename":image_filename,
                })

                progress.set_postfix({"current": generated_text[:30] + "..."})

            except Exception as e:
                print(f"\nError processing {image_filename}: {str(e)}")

            progress.update()

        progress.close()

    # 保存到 JSON 文件
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_captions, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully generated captions for {len(generated_captions)} images")
    print(f"Results saved to: {output_file}")


def run_val(args, model_type):
    output_dir = "./data/val_out"
    dataset = ClipCocoDataset(args.dataval, prefix_length=args.prefix_length, normalize_prefix=False)
    seen_ids = set()
    unique_indices = []
    for idx in range(len(dataset)):
        img_id = dataset.image_ids[idx]
        if img_id not in seen_ids:
            seen_ids.add(img_id)
            unique_indices.append(idx)

    # 使用 Subset 包装器只保留不重复的索引
    from torch.utils.data import Subset
    dataset = Subset(dataset, unique_indices)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    prefix_length = args.prefix_length
    model_path = (args.model_path)

    if args.use_lora:
        use_lora = True
    else:
        use_lora = False
    if args.use_qformer:
        use_qformer = True
    else:
        use_qformer = False
    model = ClipCaptionModelQFVE(freeze_qformer=True, prefix_length=prefix_length, clip_length=40, prefix_size=512,
                                 num_layers=12,
                                 mapping_type=MappingType.Transformer,
                                 use_qformer=use_qformer, use_lora=use_lora, llm_dim=768)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

    model.to(device)
    model.eval()

    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():  # 禁用梯度计算
        progress = tqdm(total=len(val_dataloader), desc=f"Validation {args.output_prefix}")
        generated_captions = []
        for idx, (tokens, mask, prefix) in enumerate(val_dataloader):
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            # 前向传播
            outputs = model(tokens, prefix, mask)
            if args.mapping_network == 'transformer':
                with torch.no_grad():
                    prefix_proj_test = model.clip_project(prefix[:1])
                    actual_prefix_length = prefix_proj_test.shape[1]
            else:
                actual_prefix_length = dataset.prefix_length

            logits = outputs.logits[:, actual_prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

            # 累积损失
            batch_size_actual = tokens.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

            prefix_embed = model.clip_project(prefix).reshape(prefix.size(0), actual_prefix_length, -1)

            # 为批次中的每个样本生成文本
            for i in range(prefix.size(0)):
                single_prefix_embed = prefix_embed[i:i + 1]  # 取单个样本
                generated_text = generate_beam_for_cider(
                    model, tokenizer, embed=single_prefix_embed,
                    beam_size=args.beam_size,
                    max_length=20,
                    min_length=5,
                    temperature=1.0,
                    length_penalty=0.0,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1,
                    ban_special_chars=True,
                )
                original_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
                global_idx = unique_indices[idx * val_dataloader.batch_size + i]

                generated_captions.append({
                    'image_id': original_dataset.image_ids[global_idx],
                    'caption': generated_text
                })
                # generated_captions.append({
                #     'image_id': dataset.image_ids[idx * val_dataloader.batch_size + i],
                #     'caption': generated_text
                # })
                print(generated_text)

            progress.set_postfix({"val_loss": loss.item()})
            progress.update()

        captions_file = os.path.join(output_dir, f"{args.output_prefix}_val.json")
        with open(captions_file, 'w', encoding='utf-8') as f:
            json.dump(generated_captions, f, ensure_ascii=False, indent=2)
        progress.close()
        # 计算平均损失
    avg_loss = total_loss / total_samples
    print(f"Validation completed. Average loss: {avg_loss}")
    return avg_loss


def evaluate_captions(gt_file, pred_file):
    coco = COCO(gt_file)

    # 加载你的预测结果
    with open(pred_file, 'r') as f:
        preds = json.load(f)

    # 处理重复的 image_id：每个 image_id 只保留第一个（最好的）caption
    unique_preds = {}
    for pred in preds:
        image_id = pred['image_id']
        if image_id not in unique_preds:
            unique_preds[image_id] = pred

    # 转换为列表格式
    filtered_preds = list(unique_preds.values())

    print(f"Original predictions: {len(preds)}, After filtering: {len(filtered_preds)}")

    # 构建预测结果的 COCO 格式对象
    coco_res = coco.loadRes(filtered_preds)

    # 创建评估器
    coco_eval = COCOEvalCap(coco, coco_res)

    # 获取所有 image_id
    coco_eval.params['image_id'] = coco_res.getImgIds()

    # 执行评估
    coco_eval.evaluate()

    # 打印结果
    print("=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    print("=" * 50)


def run_test(args):
    output_dir = "./data/test_out"
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载测试数据集（应该是一个只包含图像特征的数据集）
    dataset = ClipCocoDataset_test(args.dataforcap, prefix_length=40, normalize_prefix=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    prefix_length = 40
    model_path = (args.testmodel_path)
    model = ClipCaptionPrefix(prefix_length, clip_length=40, prefix_size=640, num_layers=8, mapping_type='transformer')
    model.load_state_dict(torch.load(model_path, map_location=CPU, weights_only=True))
    model = model.eval()
    model = model.to(device)

    # 创建数据加载器
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        progress = tqdm(total=len(test_dataloader), desc=f"Testing {args.output_prefix}")
        generated_captions = []

        for idx, (prefix, imageid) in enumerate(test_dataloader):
            prefix = prefix.to(device, dtype=torch.float32)

            # 投影前缀特征
            prefix_embed = model.clip_project(prefix).reshape(prefix.size(0), prefix_length, -1)

            # 为批次中的每个样本生成文本
            for i in range(prefix.size(0)):
                single_prefix_embed = prefix_embed[i:i + 1]  # 取单个样本
                generated_text = generate_beam(model, tokenizer, embed=single_prefix_embed, beam_size=3)[0]
                generated_captions.append({
                    # 'image_id': dataset.image_ids[idx * test_dataloader.batch_size + i],
                    "image_id": imageid[i].item(),
                    'caption': generated_text
                })

            progress.update()

        # 保存生成的字幕
        captions_file = os.path.join(output_dir, f"{args.output_prefix}_generated_captions.json")
        with open(captions_file, 'w', encoding='utf-8') as f:
            json.dump(generated_captions, f, ensure_ascii=False, indent=2)

        progress.close()

    print(f"Testing completed. Generated captions saved to {captions_file}")


def caption_fornocaps(args, use_beam_search, model_path, output_file):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.is_rn:
        prefix_dim = 640
        clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
    else:
        prefix_dim = 512
        clip_model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 根据模型类型设置参数
    use_lora = getattr(args, 'use_lora', False)
    use_qformer = getattr(args, 'use_qformer', False)

    model = ClipCaptionModelQFVE(prefix_length=args.prefix_length, clip_length=40, prefix_size=prefix_dim,
                                 num_layers=12,
                                 mapping_type=MappingType.Transformer, freeze_qformer=True,
                                 use_qformer=use_qformer, use_lora=use_lora, llm_dim=768)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model = model.eval()
    model = model.to(device)

    ds = load_dataset("lmms-lab/NoCaps")
    test_data = ds['test']

    print(f"Found {len(test_data)} images in NoCaps test set")

    generated_captions = []
    failed_images = []

    with torch.no_grad():
        progress = tqdm(total=len(test_data), desc="Generating captions for NoCaps test")

        for idx in range(len(test_data)):
            sample = test_data[idx]
            image_open_images_id = sample['image_open_images_id']
            image_id = sample['image_id']
            image = sample['image']  # 这是一个 PIL Image 对象

            # 如果图片加载失败或为 None
            if image is None:
                failed_images.append({"image_id": image_id, "error": "Image is None"})
                progress.update()
                continue

            try:
                # 预处理图片
                image_input = preprocess(image).unsqueeze(0).to(device)

                # 提取图像特征
                prefix = clip_model.encode_image(image_input).to(device, dtype=torch.float32)
                prefix = prefix / prefix.norm(2, -1).item()

                # 投影到模型输入维度
                prefix_embed = model.clip_project(prefix).reshape(1, args.prefix_length, -1)

                # 生成文本
                if use_beam_search:
                    generated_text = generate_beam_for_cider(
                        model, tokenizer, embed=prefix_embed,
                        beam_size=args.beam_size,
                        max_length=20,
                        min_length=8,
                        temperature=1.0,
                        length_penalty=0.2,
                        repetition_penalty=1.2,
                        ban_special_chars=True,
                    )
                else:
                    generated_text = generate2(model, tokenizer, embed=prefix_embed, top_p=0.9, temperature=0.8)

                # 保存结果
                generated_captions.append({
                    "image_id": image_id,
                    "caption": generated_text.strip(),
                })
                print(generated_text)

                progress.set_postfix({"current": generated_text[:20] + "..."})

            except Exception as e:
                print(f"\nError processing image {image_id}: {str(e)}")
                failed_images.append({
                    "image_id": image_id,
                    "error": str(e)
                })

            progress.update()

        progress.close()

    # 保存到 JSON 文件
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_captions, f, ensure_ascii=False, indent=2)

    # 保存失败记录
    if failed_images:
        failed_file = output_file.replace('.json', '_failed.json')
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_images, f, ensure_ascii=False, indent=2)
        print(f"\nFailed to process {len(failed_images)} images. Details saved to: {failed_file}")

    print(f"\nSuccessfully generated captions for {len(generated_captions)} images")
    print(f"Results saved to: {output_file}")


def ora_fornocaps(args, use_beam_search, model_path, output_file):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 根据模型类型设置参数
    use_lora = getattr(args, 'use_lora', False)
    use_qformer = getattr(args, 'use_qformer', False)
    model = ClipCaptionPrefix(prefix_length=args.prefix_length, clip_length=40, prefix_size=640, num_layers=8,
                              mapping_type=MappingType.Transformer)

    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model = model.eval()
    model = model.to(device)

    ds = load_dataset("lmms-lab/NoCaps")
    test_data = ds['test']
    # test_data = ds['validation']

    print(f"Found {len(test_data)} images in NoCaps test set")

    generated_captions = []
    failed_images = []

    with torch.no_grad():
        progress = tqdm(total=len(test_data), desc="Generating captions for NoCaps test")

        for idx in range(len(test_data)):
            sample = test_data[idx]
            image_open_images_id = sample['image_open_images_id']
            image_id = sample['image_id']
            image = sample['image']  # 这是一个 PIL Image 对象

            # 如果图片加载失败或为 None
            if image is None:
                failed_images.append({"image_id": image_id, "error": "Image is None"})
                progress.update()
                continue

            try:
                # 预处理图片
                image_input = preprocess(image).unsqueeze(0).to(device)

                # 提取图像特征
                prefix = clip_model.encode_image(image_input).to(device, dtype=torch.float32)
                prefix = prefix / prefix.norm(2, -1).item()

                # 投影到模型输入维度
                prefix_embed = model.clip_project(prefix).reshape(1, args.prefix_length, -1)

                # 生成文本
                if use_beam_search:
                    generated_text = generate_beam_for_cider(
                        model, tokenizer, embed=prefix_embed,
                        beam_size=args.beam_size,
                        max_length=20,
                        min_length=8,
                        temperature=1.0,
                        length_penalty=0.2,
                        repetition_penalty=1.2,
                        ban_special_chars=True,
                    )
                else:
                    generated_text = generate2(model, tokenizer, embed=prefix_embed, top_p=0.9, temperature=0.8)

                # 保存结果
                generated_captions.append({
                    "image_id": image_id,
                    "caption": generated_text.strip(),
                    "open_images_id": image_open_images_id
                })
                print(generated_text)

                progress.set_postfix({"current": generated_text[:20] + "..."})

            except Exception as e:
                print(f"\nError processing image {image_id}: {str(e)}")
                failed_images.append({
                    "image_id": image_id,
                    "error": str(e)
                })

            progress.update()

        progress.close()

    # 保存到 JSON 文件
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_captions, f, ensure_ascii=False, indent=2)

    # 保存失败记录
    if failed_images:
        failed_file = output_file.replace('.json', '_failed.json')
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_images, f, ensure_ascii=False, indent=2)
        print(f"\nFailed to process {len(failed_images)} images. Details saved to: {failed_file}")

    print(f"\nSuccessfully generated captions for {len(generated_captions)} images")
    print(f"Results saved to: {output_file}")

def run_flkval(args):
    output_dir = "./data/val_out"
    dataset = ClipCocoDataset(args.dataval, prefix_length=args.prefix_length, normalize_prefix=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    prefix_length = args.prefix_length
    model_path = (args.model_path)

    if args.use_lora:
        use_lora = True
    else:
        use_lora = False
    if args.use_qformer:
        use_qformer = True
    else:
        use_qformer = False
    model = ClipCaptionModelQFVE(freeze_qformer=True, prefix_length=prefix_length, clip_length=40, prefix_size=512,
                                 num_layers=12,
                                 mapping_type=MappingType.Transformer,
                                 use_qformer=use_qformer, use_lora=use_lora, llm_dim=768)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

    model.to(device)
    model.eval()

    start_idx = args.start_idx

    with open(args.dataval, 'rb') as f:
        data = pickle.load(f)

    all_clip_embeddings = data["clip_embedding"]
    all_captions_data = data["captions"]

    print(f"Total captions in dataset: {len(all_captions_data)}")
    print(f"Starting from caption index: {start_idx}")

    processed_image_ids = set()
    selected_indices = []

    for idx in range(start_idx*5, len(all_captions_data)):
        caption_item = all_captions_data[idx]
        clip_emb_idx = caption_item["clip_embedding"]
        img_id = caption_item["image_id"]

        if img_id not in processed_image_ids:
            processed_image_ids.add(img_id)
            selected_indices.append((idx, clip_emb_idx, img_id))

    print(f"Found {len(selected_indices)} unique images starting from caption {start_idx}")

    os.makedirs(output_dir, exist_ok=True)
    generated_captions = []

    with torch.no_grad():
        progress = tqdm(total=len(selected_indices), desc=f"Generating captions from index {start_idx}")

        for data_idx, clip_emb_idx, img_id in selected_indices:
            try:
                prefix = all_clip_embeddings[clip_emb_idx:clip_emb_idx+1].to(device, dtype=torch.float32)

                prefix_embed = model.clip_project(prefix)

                if hasattr(model, 'qformer_proj') or prefix_embed.size(-1) != model.gpt_embedding_size:
                    if not hasattr(model, 'qformer_proj'):
                        model.qformer_proj = torch.nn.Linear(prefix_embed.size(-1), model.gpt_embedding_size).to(device)
                    prefix_embed = model.qformer_proj(prefix_embed)

                generated_text = generate_beam_for_cider(
                    model, tokenizer, embed=prefix_embed,
                    beam_size=args.beam_size,
                    max_length=20,
                    min_length=8,
                    temperature=1.0,
                    length_penalty=0.0,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=2,
                    diversity_penalty=0,
                    ban_special_chars=True,
                )

                generated_captions.append({
                    'image_id': int(img_id),
                    'caption': generated_text
                })

                print(f"Image {img_id}: {generated_text}")

            except Exception as e:
                print(f"\nError processing image {img_id} (caption idx {data_idx}): {str(e)}")

            progress.update()

        progress.close()

    captions_file = os.path.join(output_dir, f"{args.output_prefix}_from_{start_idx}.json")
    with open(captions_file, 'w', encoding='utf-8') as f:
        json.dump(generated_captions, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully generated captions for {len(generated_captions)} images")
    print(f"Results saved to: {captions_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_model',
                        choices=('main', 'run_val', 'run_test', 'eva', 'caption_folder', 'nocaps', 'oranocaps','flickr'),
                        required=True)
    parser.add_argument('--dataval', default='./data/oscar_split_ViT-B_16_val.pkl')
    parser.add_argument('--dataora', default='./data/annotations/captions_val2017.json')  #验证集的答案
    parser.add_argument('--datatest', default='./data/val_out/ve_val_unique.json')  #被测试的数据
    parser.add_argument('--dataforcap', default='./data/oscar_split_RN50x4_test.pkl')
    parser.add_argument('--model_path', default='checkpoints/coco_prefix_trainandval_best.pt')
    parser.add_argument('--output_prefix', default="predictout")
    parser.add_argument('--prefix_length', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='clip_qf_ve',
                        choices=['clip_qf_ve', 'clip_ve', 'clip_qf', 'qf_ve', 'clip', 'qf', 've'])
    parser.add_argument('--mapping_network', default='transformer', choices=['mlp', 'transformer'])
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--use_qformer', action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--usebeamsearch', action='store_true')
    parser.add_argument('--json_file', type=str, help='Path to JSON file containing image URLs')
    parser.add_argument('--flickr_val', action='store_true')
    parser.add_argument('--start_idx', type=int, default=27000)
    args = parser.parse_args()

    modeltype = args.model_type
    if args.run_model == 'main':
        main(use_beam_search=False, model_path=args.model_path, model_type=modeltype)
    elif args.run_model == 'run_val':
        run_val(args, modeltype)
    elif args.run_model == 'run_test':
        run_test(args, modeltype)
    elif args.run_model == 'caption_folder':
        if args.usebeamsearch:
            beamsearch = True
        else:
            beamsearch = False
        caption_folder(use_beam_search=beamsearch, model_path=args.model_path,
                       image_folder=args.image_folder, output_file=args.output_file)
    elif args.run_model == 'nocaps':
        if args.usebeamsearch:
            beamsearch = True
        else:
            beamsearch = False
        caption_fornocaps(args, use_beam_search=beamsearch, model_path=args.model_path, output_file=args.output_file)
    elif args.run_model == 'oranocaps':
        if args.usebeamsearch:
            beamsearch = True
        else:
            beamsearch = False
        ora_fornocaps(args, use_beam_search=beamsearch, model_path=args.model_path, output_file=args.output_file)
    elif args.run_model == 'eva':
        evaluate_captions(args.dataora, args.datatest)
    elif args.run_model == 'flickr':
        run_flkval(args)

