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


def extract_feature(image, model, preprocess, device, model_type='clip'):
    """统一特征提取接口"""
    image_pil = Image.fromarray(image)

    if model_type == 'clip':
        image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image_tensor).cpu()
    elif model_type == 'imagenet':
        image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image_tensor).squeeze().cpu()
            if features.dim() == 1:
                features = features.unsqueeze(0)

    return features


# def main(clip_model_type: str):
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     clip_model_name = clip_model_type.replace('/', '_')
#     out_path = f"./data/oscar_split_{clip_model_name}_train.pkl"
#     clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
#     with open('../data/annotations/captions_train2017.json', 'r') as f:
#         data = json.load(f)
#     data = data["annotations"]
#     print("%0d captions loaded from json " % len(data))
#     all_embeddings = []
#     all_captions = []
#     for i in tqdm(range(len(data))):
#         d = data[i] #  {"image_id" :  , "id":   , "caption":  }
#         img_id = d["image_id"]
#         filename = f"D:/train2017/{int(img_id):012d}.jpg"
#         if not os.path.isfile(filename):
#             filename = f"D:/val2017/{int(img_id):012d}.jpg"
#         image = io.imread(filename)
#         image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
#         with torch.no_grad():
#             prefix = clip_model.encode_image(image).cpu()
#         d["clip_embedding"] = i #{"image_id" :  , "id":   , "caption":  , "clip_embedding":  }
#         #这里的应该命名clip_embedding_id更贴切
#         all_embeddings.append(prefix)
#         all_captions.append(d)
#         if (i + 1) % 10000 == 0:
#             with open(out_path, 'wb') as f:
#                 pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
#                 # {
#                 #   "clip_embedding": tensor of shape [3, 512],  # 3个图像的CLIP特征拼接
#                 #   "captions": [
#                 #     {"image_id": 123, "id": 1, "caption": "a cat", "clip_embedding": 0},
#                 #     {"image_id": 456, "id": 2, "caption": "a dog", "clip_embedding": 1},
#                 #     {"image_id": 789, "id": 3, "caption": "a bird", "clip_embedding": 2}
#                 #   ]
#                 # }
#     with open(out_path, 'wb') as f:
#         pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
#
#     print('Done')
#     print("%0d embeddings saved " % len(all_embeddings))
#     return 0
def main(clip_model_type: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 判断是使用 CLIP 模型还是 ImageNet 模型
    use_clip = clip_model_type in ['ViT-B/32', 'RN50', 'RN101', 'RN50x4']

    if use_clip:
        # 使用 CLIP 模型（经过图文对训练）
        clip_model_name = clip_model_type.replace('/', '_')
        out_path = f"../data/oscar_split_{clip_model_name}_train.pkl"
        model, preprocess = clip.load(clip_model_type, device=device, jit=False)
        model_type = 'clip'
    else:
        # 使用 ImageNet 模型（仅分类训练，无图文对训练）
        clip_model_name = clip_model_type
        out_path = f"./data/oscar_split_{clip_model_name}_train.pkl"
        model, preprocess, model_type = load_imagenet_model(clip_model_type, device)

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

        prefix = extract_feature(image, model, preprocess, device, model_type)

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
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"../data/oscar_split_{clip_model_name}_val.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('../data/annotations/captions_val2017.json', 'r') as f:
        data = json.load(f)
    data = data["annotations"]
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]  #  {"image_id" :  , "id":   , "caption":  }
        img_id = d["image_id"]
        filename = f"D:/val2017/{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            continue
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i  #{"image_id" :  , "id":   , "caption":  , "clip_embedding":  }
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 5000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
                # {
                #   "clip_embedding": tensor of shape [3, 512],  # 3个图像的CLIP特征拼接
                #   "captions": [
                #     {"image_id": 123, "id": 1, "caption": "a cat", "clip_embedding": 0},
                #     {"image_id": 456, "id": 2, "caption": "a dog", "clip_embedding": 1},
                #     {"image_id": 789, "id": 3, "caption": "a bird", "clip_embedding": 2}
                #   ]
                # }
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


def mainfortest(clip_model_type: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/oscar_split_{clip_model_name}_test.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    test_image_dir = "/home/yons/jhoge/test2017"
    all_embeddings = []
    all_images = []
    image_files = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg')]
    print("%0d images found in test directory" % len(image_files))

    for i in tqdm(range(len(image_files))):
        try:
            image_file = image_files[i]
            img_id = int(image_file.split('.')[0])  # 从文件名提取image_id
            filename = os.path.join(test_image_dir, image_file)

            # 读取并预处理图片
            image = io.imread(filename)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

            # 提取CLIP图像特征
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()

            # 保存特征和图片信息
            all_embeddings.append(prefix)
            all_images.append({
                "image_id": img_id,
                "clip_embedding": i
            })

            # 每处理5000张图片保存一次
            if (i + 1) % 5000 == 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({
                        "clip_embedding": torch.cat(all_embeddings, dim=0),
                        "images": all_images
                    }, f)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue

    # 最终保存所有数据
    with open(out_path, 'wb') as f:
        pickle.dump({
            "clip_embedding": torch.cat(all_embeddings, dim=0),
            "images": all_images
        }, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32",
                        choices=('RN50', 'RN101', 'RN50x4', "ViT-B/16", 'ViT-B/32', 'resnet50', 'resnet101'))
    parser.add_argument('--data_type', default="train", choices=('train', 'val', 'test'))
    args = parser.parse_args()
    if args.data_type == "val":
        exit(mainforval(args.clip_model_type))
    elif args.data_type == "train":
        exit(main(args.clip_model_type))
    else:
        exit(mainfortest(args.clip_model_type))
