import numpy as np
import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from PIL import ImageFile
import torchvision.models as tv_models
from torchvision import transforms
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_imagenet_model(model_name: str, device: torch.device):
    """加载仅在 ImageNet 上预训练的模型（无图文对训练）"""
    if model_name == 'resnet50':
        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        model = model.to(device)
        model.eval()
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return feature_extractor, preprocess, 'imagenet'
    elif model_name == 'resnet101':
        model = tv_models.resnet101(weights=tv_models.ResNet101_Weights.IMAGENET1K_V1)
        model = model.to(device)
        model.eval()
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return feature_extractor, preprocess, 'imagenet'
    else:
        raise ValueError(f"Unsupported ImageNet model: {model_name}")


def load_hf_vit_base_patch16(device: torch.device):
    """
    从 Hugging Face 加载 ViT-B/16 模型（仅在 ImageNet-21k 上训练，无图文对训练）

    原理说明:
    - 使用 google/vit-base-patch16 模型
    - 该模型在 ImageNet-21k (2100 万图像，21843 类) 上预训练
    - 没有经过图文对训练，只有视觉分类任务
    - 原始输出 768 维，通过投影层映射到 512 维（与 CLIP ViT-B/16 对齐）
    """
    model_path = 'google/vit-base-patch16-224'

    print(f"Loading {model_path} from Hugging Face...")

    # 加载处理器和模型
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTModel.from_pretrained(model_path)

    model = model.to(device)
    model.eval()

    # ViT-B 的隐藏层维度是 768，需要投影到 512 以匹配 CLIP ViT-B/16
    projector = nn.Linear(768, 512)
    # 初始化投影层权重
    projector.weight.data.normal_(mean=0.0, std=0.02)
    projector.bias.data.zero_()
    projector = projector.to(device)
    projector.eval()

    def preprocess(image):
        """统一的预处理接口"""
        # 确保输入是 PIL Image
        if isinstance(image, Image.Image):
            # 如果是灰度图，转换为 RGB
            if image.mode == 'L':
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
        else:
            # 如果是 numpy array，确保是 3D 的
            if image.ndim == 2:
                # 灰度图，扩展为 3 通道
                image = np.stack([image] * 3, axis=-1)
            # 转换为 PIL Image
            image = Image.fromarray(image)
            inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        return pixel_values

    return model, projector, preprocess, 'imagenet'


def extract_feature(image, model_type, clip_model=None, hf_model=None,
                    projector=None, preprocess=None, device=None):
    """
    统一的特征提取接口

    Args:
        image: numpy array 格式的图像
        model_type: 'clip' 或 'imagenet'
        clip_model: CLIP 模型对象
        hf_model: (model, projector) 元组，Hugging Face ViT 模型
        preprocess: 对应的预处理函数
        device: torch device

    Returns:
        features: 提取的特征向量
    """
    image_pil = Image.fromarray(image)

    if model_type == 'clip':
        # CLIP 模型的预处理和特征提取
        image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(image_tensor).cpu()

    elif model_type == 'imagenet':
        # Hugging Face ViT 模型的特征提取
        model, proj = hf_model

        # 预处理
        inputs = preprocess(image_pil)
        if isinstance(inputs, dict):
            pixel_values = inputs['pixel_values'].to(device)
        else:
            pixel_values = inputs.unsqueeze(0).to(device)

        with torch.no_grad():
            # ViTModel 输出包含 last_hidden_state
            outputs = model(pixel_values=pixel_values)
            # 获取 [CLS] token 的特征 (第一个 token)
            cls_features = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
            # 投影到 512 维（与 CLIP ViT-B/16 对齐）
            features = proj(cls_features).cpu()

    return features


def main(clip_model_type: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 判断是使用 CLIP 模型还是 ImageNet 模型
    use_clip = clip_model_type in ['ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'ViT-B/16']

    # Hugging Face ViT 模型列表
    hf_vit_models = ['vit-base-patch16']
    use_hf_vit = clip_model_type in hf_vit_models

    if use_clip:
        # 使用 CLIP 模型（经过图文对训练）
        clip_model_name = clip_model_type.replace('/', '_')
        out_path = f"../data/oscar_split_{clip_model_name}_train.pkl"
        model, preprocess = clip.load(clip_model_type, device=device, jit=False)
        model_type = 'clip'
        projector = None
        print(f"Using CLIP {clip_model_type} (WITH language-image pretraining)")
    elif use_hf_vit:
        # 使用 Hugging Face ViT 模型（仅 ImageNet 分类，无图文对训练）
        clip_model_name = clip_model_type.replace('-', '_')
        out_path = f"../data/oscar_split_{clip_model_name}_train.pkl"
        model, projector, preprocess, model_type = load_hf_vit_base_patch16(device)
        print(f"Using HF {clip_model_type} (WITHOUT language-image pretraining)")
        print("Note: This model is trained on ImageNet-21k classification only")
    else:
        # 使用 ImageNet ResNet 模型
        clip_model_name = clip_model_type
        out_path = f"./data/oscar_split_{clip_model_name}_train.pkl"
        model, preprocess, model_type = load_imagenet_model(clip_model_type, device)
        projector = None

    with open('../data/annotations/captions_train2017.json', 'r') as f:
        data = json.load(f)
    data = data["annotations"]
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = f"D:/train2017/{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"D:/val2017/{int(img_id):012d}.jpg"
        image = io.imread(filename)
        if use_clip:
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = model.encode_image(image).cpu()
        elif use_hf_vit:
            prefix = extract_feature(
                image,
                model_type,
                hf_model=(model, projector),
                preprocess=preprocess,
                device=device
            )
        else:
            prefix = extract_feature(
                image,
                model_type,
                clip_model=model if use_clip else None,
                imagenet_model=model if not use_clip else None,
                projector=projector,
                preprocess=preprocess,
                device=device
            )

        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    print(f"Model type: {model_type} (has language-image pretraining: {use_clip})")
    return 0


def mainforval(clip_model_type: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 判断是使用 CLIP 模型还是 ImageNet 模型
    use_clip = clip_model_type in ['ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'ViT-B/16']
    hf_vit_models = ['vit-base-patch16']
    use_hf_vit = clip_model_type in hf_vit_models

    if use_clip:
        # 使用 CLIP 模型（经过图文对训练）
        clip_model_name = clip_model_type.replace('/', '_')
        out_path = f"../data/oscar_split_{clip_model_name}_val.pkl"
        model, preprocess = clip.load(clip_model_type, device=device, jit=False)
        model_type = 'clip'
        projector = None
        print(f"Using CLIP {clip_model_type} (WITH language-image pretraining)")
    elif use_hf_vit:
        # 使用 Hugging Face ViT 模型（仅 ImageNet 分类，无图文对训练）
        clip_model_name = clip_model_type.replace('-', '_')
        out_path = f"../data/oscar_split_{clip_model_name}_val.pkl"
        model, projector, preprocess, model_type = load_hf_vit_base_patch16(device)
        print(f"Using HF {clip_model_type} (WITHOUT language-image pretraining)")
        print("Note: This model is trained on ImageNet-21k classification only")
    else:
        # 使用 ImageNet ResNet 模型
        clip_model_name = clip_model_type
        out_path = f"./data/oscar_split_{clip_model_name}_val.pkl"
        model, preprocess, model_type = load_imagenet_model(clip_model_type, device)
        projector = None

    with open('../data/annotations/captions_val2017.json', 'r') as f:
        data = json.load(f)
    data = data["annotations"]
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = f"D:/val2017/{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            continue
        image = io.imread(filename)
        if use_clip:
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = model.encode_image(image).cpu()
        elif use_hf_vit:
            prefix = extract_feature(
                image,
                model_type,
                hf_model=(model, projector),
                preprocess=preprocess,
                device=device
            )
        else:
            prefix = extract_feature(
                image,
                model_type,
                clip_model=model if use_clip else None,
                imagenet_model=model if not use_clip else None,
                projector=projector,
                preprocess=preprocess,
                device=device
            )

        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 5000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    print(f"Model type: {model_type} (has language-image pretraining: {use_clip})")
    return 0


def mainfortest(clip_model_type: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    use_clip = clip_model_type in ['ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'ViT-B/16']
    hf_vit_models = ['vit-base-patch16']
    use_hf_vit = clip_model_type in hf_vit_models

    if use_clip:
        # 使用 CLIP 模型（经过图文对训练）
        clip_model_name = clip_model_type.replace('/', '_')
        out_path = f"./data/oscar_split_{clip_model_name}_test.pkl"
        model, preprocess = clip.load(clip_model_type, device=device, jit=False)
        model_type = 'clip'
        projector = None
        print(f"Using CLIP {clip_model_type} (WITH language-image pretraining)")
    elif use_hf_vit:
        # 使用 Hugging Face ViT 模型（仅 ImageNet 分类，无图文对训练）
        clip_model_name = clip_model_type.replace('-', '_')
        out_path = f"./data/oscar_split_{clip_model_name}_test.pkl"
        model, projector, preprocess, model_type = load_hf_vit_base_patch16(device)
        print(f"Using HF {clip_model_type} (WITHOUT language-image pretraining)")
        print("Note: This model is trained on ImageNet-21k classification only")
    else:
        # 使用 ImageNet ResNet 模型
        clip_model_name = clip_model_type
        out_path = f"./data/oscar_split_{clip_model_name}_test.pkl"
        model, preprocess, model_type = load_imagenet_model(clip_model_type, device)
        projector = None

    test_image_dir = "/home/yons/jhoge/test2017"
    all_embeddings = []
    all_images = []
    image_files = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg')]
    print("%0d images found in test directory" % len(image_files))

    for i in tqdm(range(len(image_files))):
        try:
            image_file = image_files[i]
            img_id = int(image_file.split('.')[0])
            filename = os.path.join(test_image_dir, image_file)

            image = io.imread(filename)

            if use_hf_vit:
                prefix = extract_feature(
                    image,
                    model_type,
                    hf_model=(model, projector),
                    preprocess=preprocess,
                    device=device
                )
            else:
                image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
                with torch.no_grad():
                    prefix = model.encode_image(image).cpu()

            all_embeddings.append(prefix)
            all_images.append({
                "image_id": img_id,
                "clip_embedding": i
            })

            if (i + 1) % 5000 == 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({
                        "clip_embedding": torch.cat(all_embeddings, dim=0),
                        "images": all_images
                    }, f)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue

    with open(out_path, 'wb') as f:
        pickle.dump({
            "clip_embedding": torch.cat(all_embeddings, dim=0),
            "images": all_images
        }, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    print(f"Model type: {model_type} (has language-image pretraining: {use_clip})")
    return 0


def flickr_func(clip_model_type: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 判断是使用 CLIP 模型还是 ImageNet 模型
    use_clip = clip_model_type in ['ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'ViT-B/16']

    # Hugging Face ViT 模型列表
    hf_vit_models = ['vit-base-patch16']
    use_hf_vit = clip_model_type in hf_vit_models

    if use_clip:
        # 使用 CLIP 模型（经过图文对训练）
        clip_model_name = clip_model_type.replace('/', '_')
        out_path = f"../data/oscar_split_{clip_model_name}_trainflk.pkl"
        model, preprocess = clip.load(clip_model_type, device=device, jit=False)
        model_type = 'clip'
        projector = None
        print(f"Using CLIP {clip_model_type} (WITH language-image pretraining)")
    elif use_hf_vit:
        # 使用 Hugging Face ViT 模型（仅 ImageNet 分类，无图文对训练）
        clip_model_name = clip_model_type.replace('-', '_')
        out_path = f"../data/oscar_split_{clip_model_name}_train.pkl"
        model, projector, preprocess, model_type = load_hf_vit_base_patch16(device)
        print(f"Using HF {clip_model_type} (WITHOUT language-image pretraining)")
        print("Note: This model is trained on ImageNet-21k classification only")
    else:
        # 使用 ImageNet ResNet 模型
        clip_model_name = clip_model_type
        out_path = f"./data/oscar_split_{clip_model_name}_train.pkl"
        model, preprocess, model_type = load_imagenet_model(clip_model_type, device)
        projector = None

    with open('../data/annotations/flickr.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data = data["annotations"]
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    # for i in tqdm(range(len(data))):
    for num, i in tqdm(enumerate(tqdm(data))):
        # d = data[i]
        d = []

        # img_id = d["image_id"]
        img_id = i[0:-4]
        for temp in data[i]:
            d.append({"image_id": int(img_id), "caption": temp})

        filename = f"../flickr30k_images/{i}"
        if not os.path.isfile(filename):
            # filename = f"D:/val2017/{int(img_id):012d}.jpg"
            continue
        image = io.imread(filename)
        if use_clip:
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = model.encode_image(image).cpu()
        elif use_hf_vit:
            prefix = extract_feature(
                image,
                model_type,
                hf_model=(model, projector),
                preprocess=preprocess,
                device=device
            )
        else:
            prefix = extract_feature(
                image,
                model_type,
                clip_model=model if use_clip else None,
                imagenet_model=model if not use_clip else None,
                projector=projector,
                preprocess=preprocess,
                device=device
            )
        for temp in d:
            temp["clip_embedding"] = num
        # d["clip_embedding"] = num
        all_embeddings.append(prefix)
        all_captions = d + all_captions
        if (num + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    print(f"Model type: {model_type} (has language-image pretraining: {use_clip})")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32",
                        choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/16', 'ViT-B/32',
                                 'resnet50', 'resnet101', 'vit-base-patch16'))
    parser.add_argument('--data_type', default="train", choices=('train', 'val', 'test', 'flickr'))
    args = parser.parse_args()
    if args.data_type == "val":
        exit(mainforval(args.clip_model_type))
    elif args.data_type == "train":
        exit(main(args.clip_model_type))
    elif args.data_type == "test":
        exit(mainfortest(args.clip_model_type))
    elif args.data_type == "flickr":
        exit(flickr_func(args.clip_model_type))
