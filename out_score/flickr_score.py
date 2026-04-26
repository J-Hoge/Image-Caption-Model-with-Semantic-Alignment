import json
import numpy as np
import argparse
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def _extract_score(score):
    """安全提取标量分数（兼容 list 或 float 返回格式）"""
    if isinstance(score, (list, tuple)):
        return float(score[0])
    return float(score)


def evaluate_captions(gt_dict, pred_list, id_mapping=None):
    # 1. 数据对齐
    pred_dict = {}
    for p in pred_list:
        raw_id = p.get("image_id")
        raw_id= str(raw_id) + ".jpg"
        # 统一转为字符串匹配，若提供映射表则优先使用
        gt_key = id_mapping.get(raw_id, str(raw_id)) if id_mapping else str(raw_id)
        if gt_key in gt_dict:
            pred_dict[gt_key] = p["caption"]

    common_keys = set(gt_dict.keys()) & set(pred_dict.keys())
    if not common_keys:
        raise ValueError("❌ 预测结果与答案的 image_id 未匹配，请检查数据格式或提供 id_mapping。")

    print(f"✅ 成功匹配 {len(common_keys)} 张图片")

    scorers = {
        "bleu": Bleu(4),
        "meteor": Meteor(),
        "rouge": Rouge(),
        "cider": Cider()
    }
    tokenizer = PTBTokenizer()

    img_scores = {k: [] for k in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "CIDEr", "ROUGE-L"]}

    # 2. 构建全局CIDEr评估数据
    gts_for_cider = {}
    res_for_cider = {}
    for img_id in common_keys:
        gts_for_cider[img_id] = [{"caption": ref} for ref in gt_dict[img_id]]
        res_for_cider[img_id] = [{"caption": pred_dict[img_id]}]

    gts_tokenized = tokenizer.tokenize(gts_for_cider)
    res_tokenized = tokenizer.tokenize(res_for_cider)
    cider_scores_all, _ = scorers["cider"].compute_score(gts_tokenized, res_tokenized)
    if isinstance(cider_scores_all, (list, np.ndarray)):
        cider_score_dict = dict(zip(common_keys, cider_scores_all))
    else:
        cider_score_dict = {img_id: float(cider_scores_all) for img_id in common_keys}

    # 3. 逐图对比其他指标
    for img_id in tqdm(common_keys, desc="Evaluating"):
        pred_cap = pred_dict[img_id]
        ref_caps = gt_dict[img_id]
        best_scores = {k: 0.0 for k in img_scores.keys()}

        for ref in ref_caps:
            # pycocoevalcap 标准输入格式: {image_id: [caption]}
            gts = {img_id: [ref]}
            res = {img_id: [pred_cap]}

            # BLEU-1~4
            bleu_score, _ = scorers["bleu"].compute_score(gts, res)
            for i in range(4):
                if bleu_score[i] > best_scores[f"Bleu_{i + 1}"]:
                    best_scores[f"Bleu_{i + 1}"] = bleu_score[i]

            # METEOR
            m_val = _extract_score(scorers["meteor"].compute_score(gts, res)[0])
            if m_val > best_scores["METEOR"]:
                best_scores["METEOR"] = m_val

            # ROUGE-L
            r_val = _extract_score(scorers["rouge"].compute_score(gts, res)[0])
            if r_val > best_scores["ROUGE-L"]:
                best_scores["ROUGE-L"] = r_val

        # 直接使用全局CIDEr分数
        best_scores["CIDEr"] = cider_score_dict[img_id]

        for k in img_scores:
            img_scores[k].append(best_scores[k])

    # 4. 计算全局平均分（转为百分制）
    final_scores = {k: float(np.mean(v)) * 100 for k, v in img_scores.items()}
    return final_scores


if __name__ == "__main__":
    # ================= 配置路径 =================
    GT_PATH = "../data/annotations/flickr.json"  # 替换为你的 GT 文件路径
    PRED_PATH = "../data/val_out/flickrout_from_27001.json"  # 替换为你的预测文件路径
    MAP_PATH = None  # 可选: ID映射文件路径，例如 {"4500": "1000092795.jpg"}
    # ============================================

    print("📂 正在加载 JSON 文件...")
    try:
        with open(GT_PATH, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(PRED_PATH, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        print(f"✅ GT 加载成功: {len(gt_data)} 张图片")
        print(f"✅ 预测加载成功: {len(pred_data)} 条结果")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"❌ 找不到文件: {e.filename}。请检查路径配置。")
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ JSON 解析失败: {e}")

    # 加载可选的 ID 映射表
    id_mapping = None
    if MAP_PATH:
        with open(MAP_PATH, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
        print(f"✅ ID 映射表已加载")

    # 运行评估
    results = evaluate_captions(gt_data, pred_data, id_mapping=id_mapping)

    # 打印 & 保存结果
    print("\n📊 最终评估结果 (百分制):")
    print(json.dumps(results, indent=4, ensure_ascii=False))

    out_path = "caption_eval_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"💾 结果已保存至: {out_path}")
