import json
import numpy as np
from collections import defaultdict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from torchmetrics.functional.multimodal import clip_score
from torchvision.transforms import ToTensor
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from rouge_score import rouge_scorer
from PIL import Image
import os


def load_json(json_path):
    """加载JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def prepare_coco_format(gt_data, pred_data):
    """
    将数据转换为COCO评估格式
    参数:
        gt_data: 真实标注数据列表
        pred_data: 预测结果数据列表
    返回: gts(真实标签), res(预测结果)
    """
    gts = defaultdict(list)
    res = defaultdict(list)

    for item in gt_data:
        image_id = item['image_id']
        caption = item['caption']
        gts[image_id].append({'caption': caption})

    seen_images = set()
    for item in pred_data:
        image_id = item['image_id']
        if image_id not in seen_images:
            caption = item['caption']
            res[image_id].append({'caption': caption})
            seen_images.add(image_id)

    return gts, res



def compute_bleu_meteor_cider_spice(gts, res):
    """
    计算BLEU-4, METEOR, CIDEr, SPICE指标
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
    ]

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    final_scores = {}

    print('Computing ROUGE-L scores...')
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    for image_id in gts.keys():
        if image_id in res:
            target = gts[image_id]
            prediction_str = res[image_id]
            if isinstance(target, list):
                target = target[0]
            if isinstance(prediction_str, list):
                prediction_str = prediction_str[0]
            scores = scorer.score(target, prediction_str)
            rouge_scores.append(scores['rougeL'].fmeasure)
    if rouge_scores:
        avg_rouge_l = np.mean(rouge_scores)
        final_scores['ROUGE-L'] = avg_rouge_l
        print(f'ROUGE-L: {avg_rouge_l:.4f}')
    
    for scorer, method in scorers:
        print(f'Computing {method} scores...')
        score, scores = scorer.compute_score(gts, res)

        if isinstance(method, list):
            for m, s in zip(method, score):
                final_scores[m] = s
                print(f'{m}: {s:.4f}')
        else:
            final_scores[method] = score
            print(f'{method}: {score:.4f}')




    return final_scores


def compute_clipscore(pred_data, image_dir=None):
    """
    计算CLIPScore指标
    需要提供图像目录路径
    """
    # 去重获取唯一的image_id和对应的caption
    image_captions = {}
    for item in pred_data:
        image_id = item['image_id']
        if image_id not in image_captions:
            image_captions[image_id] = item['caption']

    scores = []

    for image_id, caption in image_captions.items():
        try:
            if image_dir:
                img_path = os.path.join(image_dir, f'COCO_val2014_{image_id:012d}.jpg')
            else:
                continue

            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert('RGB')
            image_tensor = ToTensor()(image).unsqueeze(0)

            score = clip_score(
                image_tensor,
                [caption],
                model_name_or_path="openai/clip-vit-base-patch16"
            )
            scores.append(score.item())

        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue

    if scores:
        avg_clipscore = np.mean(scores)
        print(f'CLIPScore: {avg_clipscore:.4f}')
        return avg_clipscore
    else:
        print("No valid CLIPScore computed")
        return None


def evaluate_all(gt_json_path, pred_json_path, image_dir=None):
    """
    综合评估所有指标
    参数:
        gt_json_path: 真实标注JSON文件路径
        pred_json_path: 预测结果JSON文件路径
        image_dir: 图像目录路径（用于CLIPScore）
    """
    print("=" * 50)
    print("Starting Evaluation...")
    print("=" * 50)

    gt_data = load_json(gt_json_path)
    pred_data = load_json(pred_json_path)

    gts, res = prepare_coco_format(gt_data, pred_data)

    print(f"Total images: {len(gts)}")
    print(f"Total predictions: {len(res)}")
    print("-" * 50)

    text_scores = compute_bleu_meteor_cider_spice(gts, res)

    print("-" * 50)

    if image_dir:
        clipscore = compute_clipscore(pred_data, image_dir)
        if clipscore:
            text_scores['CLIPScore'] = clipscore

    print("=" * 50)
    print("Final Results:")
    print("=" * 50)
    for metric, score in text_scores.items():
        print(f"{metric}: {score:.4f}")

    return text_scores


if __name__ == "__main__":
    gt_json = r"D:\python_code\clipcap\data\annotations\cocoval2017.json"
    pred_json = r"D:\python_code\clipcap\data\val_out\val_out\clipqfve_val.json"

    image_directory = None

    scores = evaluate_all(gt_json, pred_json, image_directory)
