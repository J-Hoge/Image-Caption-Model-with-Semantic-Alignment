import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Optional
from transformers import GPT2Tokenizer
import sys
import os

# 添加 clipcap 目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from clip_qf_train import ClipCaptionModelQFVE
from predict_ve import generate_beam_for_cider

class GradCAMVisualizer:
    """
    基于 Grad-CAM 的注意力可视化 - 针对 Q-Former + GPT-2 架构优化

    原理说明:
    1. Grad-CAM 利用目标类别的梯度对最后一层卷积/Transformer 层的特征图加权
    2. 对于 VQA/Image Captioning 模型，梯度从生成的文本反向传播到图像特征
    3. Q-Former 的 cross-attention 层是理想的可视化目标（连接视觉和语言）
    """

    def __init__(self, model: ClipCaptionModelQFVE, clip_model, tokenizer, device=None):
        """
        Args:
            model: 训练好的 ClipCaptionModelQFVE 模型
            clip_model: CLIP 图像编码器
            tokenizer: GPT-2 tokenizer
            device: 计算设备
        """
        self.model = model.eval()
        self.clip_model = clip_model.eval()
        self.tokenizer = tokenizer
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将模型移到指定设备
        self.model = self.model.to(self.device).float()
        self.clip_model = self.clip_model.to(self.device).float()

        # 注册 hook 用于获取梯度和特征
        self.clip_feature = None
        self.clip_gradient = None
        self.qformer_attention = None

        # 为 CLIP 的视觉编码器注册 hook
        self._register_clip_hooks()

        # 为 Q-Former 注册 hook（可选，用于对比）
        self._register_qformer_hooks()

    def _register_clip_hooks(self):
        """为 CLIP 的视觉编码器注册前向和反向 hook"""
        clip_visual = self.clip_model.visual

        def forward_hook(module, input, output):
            # CLIP ViT 输出：[B, N, D] - N 是 patch 数量 + 1 (CLS token)
            self.clip_feature = output.permute(1, 0, 2).detach()

        def backward_hook(module, grad_input, grad_output):
            self.clip_gradient = grad_output[0].permute(1, 0, 2).detach()


        # 注册到 CLIP 的最后一个 transformer 层
        # CLIP ViT-B/16: 12 层 transformer, 每层有 self-attention
        if hasattr(clip_visual, 'transformer') and hasattr(clip_visual.transformer, 'resblocks'):
            last_resblock = clip_visual.transformer.resblocks[-1]
            last_resblock.register_forward_hook(forward_hook)
            last_resblock.register_full_backward_hook(backward_hook)
        elif hasattr(clip_visual, 'ln_post'):
            # 如果是 post-layernorm 输出
            clip_visual.ln_post.register_forward_hook(forward_hook)
            clip_visual.ln_post.register_full_backward_hook(backward_hook)

    def _register_qformer_hooks(self):
        """为 Q-Former 的 cross-attention 层注册 hook"""
        if not hasattr(self.model.clip_project, 'qformer_module'):
            return

        qformer = self.model.clip_project.qformer_module.qformer

        def qformer_forward_hook(module, input, output):
            # Q-Former 输出：[B, 32, 768] - 32 个 query tokens
            if isinstance(output, tuple):
                self.qformer_attention = output[0].detach()
            else:
                self.qformer_attention = output.detach()

        # 注册到 Q-Former 的最后一层
        if hasattr(qformer, 'encoder') and hasattr(qformer.encoder, 'layer'):
            last_layer = qformer.encoder.layer[-1]
            last_layer.register_forward_hook(qformer_forward_hook)

    def generate_cam(self, image: Image.Image, target_text: str = None,
                     use_generated_text: bool = True) -> Tuple[np.ndarray, str]:
        """
        生成 Grad-CAM 热力图

        核心步骤:
        1. 前向传播：图像 → CLIP → Q-Former → GPT-2 → 生成文本
        2. 计算 loss：基于生成文本的交叉熵
        3. 反向传播：loss → GPT-2 → Q-Former → CLIP
        4. 提取 CLIP 最后一层的梯度，计算 Grad-CAM

        Args:
            image: PIL 图像
            target_text: 目标文本 (如果不指定则使用生成的文本)
            use_generated_text: 是否使用模型生成的文本

        Returns:
            cam: 归一化的热力图 [H, W]
            generated_text: 生成的文本
        """
        # 1. 图像预处理
        image_tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)
        original_image = np.array(image)
        original_h, original_w = original_image.shape[:2]

        # 2. 前向传播（启用梯度）
        image_tensor.requires_grad_(True)

        with torch.enable_grad():
            # 获取 CLIP 图像特征 [B, 50, 512] (ViT-B/16: 14x14 patches + CLS)
            clip_features = self.clip_model.encode_image(image_tensor)
            clip_features = clip_features.float()

            # 通过 Q-Former 获取 prefix [B, 32, 768]
            prefix_embeddings = self.model.clip_project(clip_features)

            # 维度投影（如果需要）
            if hasattr(prefix_embeddings, 'size'):
                if prefix_embeddings.size(-1) != self.model.gpt_embedding_size:
                    if not hasattr(self.model, 'qformer_proj'):
                        self.model.qformer_proj = nn.Linear(
                            prefix_embeddings.size(-1),
                            self.model.gpt_embedding_size
                        )
                    prefix_embeddings = self.model.qformer_proj(prefix_embeddings)

            # 3. 生成文本
            if use_generated_text or target_text is None:
                generated_ids = self._generate_greedy(prefix_embeddings, max_length=30)
                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            else:
                generated_text = target_text
            generated_text = generate_beam_for_cider(
                model, tokenizer, embed=prefix_embeddings,
                beam_size=5,
                max_length=20,
                min_length=8,
                temperature=1.0,
                length_penalty=0.2,
                repetition_penalty=1.2,
                ban_special_chars=True,
            )
            # 4. 计算生成文本的 loss（用于梯度反向传播）
            target_tokens = self.tokenizer.encode(generated_text, return_tensors='pt').to(self.device)

            # 拼接 prefix 和文本
            prefix_len = prefix_embeddings.shape[1]
            token_embeds = self.model.gpt.transformer.wte(target_tokens)
            inputs_embeds = torch.cat([prefix_embeddings, token_embeds], dim=1)

            # 创建 attention mask
            attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long).to(self.device)

            # 创建 labels（prefix 部分设为 -100 忽略）
            labels = torch.cat([
                torch.full((1, prefix_len), -100).to(self.device),
                target_tokens
            ], dim=1)

            # 前向传播计算 loss
            outputs = self.model.gpt(
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=attention_mask
            )
            loss = outputs.loss

        # 5. 反向传播获取梯度
        self.model.zero_grad()
        self.clip_model.zero_grad()
        loss.backward()


        if self.clip_gradient is None or self.clip_feature is None:
            raise RuntimeError("无法获取 CLIP 梯度，请检查 hook 是否正确注册")

        # 打印梯度信息用于调试
        print(f"CLIP feature shape: {self.clip_feature.shape}")
        print(f"CLIP gradient shape: {self.clip_gradient.shape}")
        print(f"Gradient norm: {self.clip_gradient.norm().item()}")


        # 7. 计算 Grad-CAM
        pooled_gradients = self._compute_grad_cam(self.clip_gradient)
        cam = self._compute_cam(self.clip_feature, pooled_gradients)

        # 8. 将 CAM 映射到原图大小
        cam_resized = self._resize_cam(cam, (original_h, original_w))
        cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)

        return cam_normalized, generated_text

    def _compute_grad_cam(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        计算 Grad-CAM 的权重

        对空间维度（patch 数量）进行全局平均池化
        gradients: [B, N, D] -> pooled_gradients: [B, D]
        """
        # 全局平均池化：对 N（patch 维度）求平均
        pooled_gradients = torch.mean(gradients, dim=1)  # [B, D]
        return pooled_gradients

    def _compute_cam(self, features: torch.Tensor, weights: torch.Tensor) -> np.ndarray:

        batch_size = features.shape[0]

        # 加权求和：[B, D] × [B, N, D] -> [B, N]
        # 使用 einsum 更高效
        cam = torch.einsum('bd,bnd->bn', weights, features)

        # 转换为 numpy
        cam_np = cam[0].detach().cpu().numpy()  # [N]

        # CLIP ViT-B/16: N=50 (14x14 patches + 1 CLS token)
        # 需要将 1D 序列转换为 2D 网格
        n_patches = len(cam_np)

        # 尝试推断网格大小
        # CLIP ViT-B/16: image_size=224, patch_size=16 -> 14x14 = 196 patches
        # 但实际输出可能是下采样或其他配置
        try:
            # 假设有 CLS token
            if int(np.sqrt(n_patches - 1)) ** 2 == n_patches - 1:
                grid_size = int(np.sqrt(n_patches - 1))
                # 移除 CLS token 并 reshape
                cam_2d = cam_np[1:].reshape(grid_size, grid_size)
            # 假设没有 CLS token
            elif int(np.sqrt(n_patches)) ** 2 == n_patches:
                grid_size = int(np.sqrt(n_patches))
                cam_2d = cam_np.reshape(grid_size, grid_size)
            else:
                # 无法 reshape，保持 1D
                cam_2d = cam_np
        except:
            cam_2d = cam_np

        return cam_2d


    def _resize_cam(self, cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        将 CAM 热力图 resize 到原图大小

        使用双线性插值将低分辨率热力图（如 14x14）上采样到原图尺寸
        """
        if cam.ndim == 1:
            # 如果是 1D，先转换为 2D（假设是正方形）
            grid_size = int(np.sqrt(len(cam)))
            if grid_size * grid_size == len(cam):
                cam = cam.reshape(grid_size, grid_size)
            else:
                # 无法转换，返回原图大小的均匀分布
                return np.ones(target_size) * 0.5

        # 确保 cam 不是空数组
        if cam.size == 0 or cam.shape[0] == 0 or cam.shape[1] == 0:
            return np.ones(target_size) * 0.5

        # 使用 OpenCV 双线性插值
        cam_resized = cv2.resize(cam, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        return cam_resized

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """预处理图像以适配 CLIP"""
        preprocess = clip.load("ViT-B/16", device=self.device)[1]
        return preprocess(image)

    def _generate_greedy(self, prefix_embeds: torch.Tensor, max_length: int = 30) -> torch.Tensor:
        """使用贪婪搜索生成文本"""
        device = prefix_embeds.device
        generated = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long, device=device)

        for _ in range(max_length):
            with torch.no_grad():
                token_embed = self.model.gpt.transformer.wte(generated)
                inputs_embeds = torch.cat([prefix_embeds, token_embed], dim=1)
                attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=device)

                outputs = self.model.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def visualize(self, image_path: str, save_path: Optional[str] = None,
                  show: bool = True) -> Tuple[np.ndarray, str]:
        """
        可视化 Grad-CAM 结果

        输出三列:
        1. 原始图像
        2. 热力图叠加 + 生成文本
        3. 纯热力图
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)

        # 生成 CAM
        cam, generated_text = self.generate_cam(image)

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # 1. 原始图像
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 2. 热力图叠加
        # 将 CAM 转换为彩色热力图 (JET colormap)
        cam_colored = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

        # 叠加到原图 (alpha blending)
        overlay = cv2.addWeighted(original_image, 0.6, cam_colored, 0.4, 0)
        axes[1].imshow(overlay)
        axes[1].set_title(f'Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # 3. 纯热力图
        im = axes[2].imshow(cam, cmap='jet')
        axes[2].set_title(f'CAM (Attention Map)', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="3%", pad=0.08)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=9)

        plt.tight_layout(w_pad=2.5)

        # 保存图像
        if save_path:
            # 分别保存三张图
            base_path = save_path.replace('.jpg', '')

            # 1. 原始图像
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.imshow(original_image)
            ax1.axis('off')
            plt.tight_layout()
            fig1.savefig(f'{base_path}_original.jpg', dpi=300, bbox_inches='tight')
            plt.close(fig1)

            # 2. 热力图叠加
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.imshow(overlay)
            ax2.set_title(f'', fontsize=14, fontweight='bold')
            ax2.axis('off')
            plt.tight_layout()
            fig2.savefig(f'{base_path}_heatmap.jpg', dpi=300, bbox_inches='tight')
            plt.close(fig2)

            # 3. 纯热力图
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            im = ax3.imshow(cam, cmap='jet')
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="3%", pad=0.08)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=9)
            ax3.axis('off')
            plt.tight_layout()
            fig3.savefig(f'{base_path}_cam.jpg', dpi=300, bbox_inches='tight')
            plt.close(fig3)

            # 同时保留合并图
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到：{base_path}_original.jpg, {base_path}_heatmap.jpg, {base_path}_cam.jpg")

            # 显示
        if show:
            plt.show()

        plt.close()

        return cam, generated_text


def compare_methods(image_path: str, model_paths: List[dict], save_path: Optional[str] = None):
    """
    对比不同方法的可视化结果（类似论文中的对比图）

    Args:
        image_path: 图像路径
        model_paths: 模型配置列表，每个元素包含:
            - model_path: 模型权重路径
            - model_name: 模型名称 (如 "CLIP", "Ours")
            - use_qformer: 是否使用 Q-Former
            - use_lora: 是否使用 LoRA
        save_path: 保存路径
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 CLIP 和 tokenizer
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 加载原始图像
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)

    # 创建图表
    num_methods = len(model_paths) + 1  # +1 for original image
    fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 5))

    if num_methods == 1:
        axes = [axes]

    # 1. 显示原始图像
    axes[0].imshow(original_image)
    axes[0].set_title('Image', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # 2. 对每个模型生成可视化
    for i, model_config in enumerate(model_paths):
        model_path = model_config['model_path']
        model_name = model_config['model_name']
        use_qformer = model_config.get('use_qformer', False)
        use_lora = model_config.get('use_lora', False)

        # 加载模型
        model = ClipCaptionModelQFVE(
            prefix_length=32,
            clip_length=40,
            prefix_size=512,
            use_qformer=use_qformer,
            use_lora=use_lora,
            llm_dim=768
        )
        model.load_state_dict(torch.load(model_path, map_location=device))

        # 创建可视化器
        visualizer = GradCAMVisualizer(model, clip_model, tokenizer, device)

        # 生成 CAM
        cam, generated_text = visualizer.generate_cam(image)

        # 创建热力图叠加
        cam_colored = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_image, 0.6, cam_colored, 0.4, 0)

        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f'{model_name}\n{generated_text}', fontsize=12, fontweight='bold')
        axes[i + 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到：{save_path}")

    plt.show()
    plt.close()


# 使用示例
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagepath',type=str,default='./Images/000000224051.jpg')
    parser.add_argument('--modelpath',type=str,default='./checkpoints/base/.pt')
    parser.add_argument('--savedir', type=str,default='.')
    parser.add_argument('--useqformer',action='store_true')
    parser.add_argument('--uselora',action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_path = args.imagepath

    print(f"Using device: {device}")
    if args.useqformer:
        useqformer = True
    else:
        useqformer = False
    if args.uselora:
        uselora = True
    else:
        uselora = False
    # 加载模型
    model = ClipCaptionModelQFVE( 
        prefix_length=32,
        clip_length=40,
        prefix_size=512,
        use_qformer=useqformer,
        use_lora=uselora,
        llm_dim=768,
        freeze_qformer=True,
    )

    try:
        model.load_state_dict(torch.load(args.modelpath, map_location=device), strict=False)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Warning: Model file not found at {args.modelpath}")
        print("Please update the model_path to your trained model.")
        # 创建一个示例模型继续运行
        use_qformer = False
        use_lora = False
        model = ClipCaptionModelQFVE(

            prefix_length=32,
            clip_length=40,
            prefix_size=512,
            use_qformer=use_qformer,
            use_lora=use_lora,
            llm_dim=768
        )

    # 加载 CLIP 和 tokenizer
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 创建可视化器
    visualizer = GradCAMVisualizer(model, clip_model, tokenizer, device)

    # 可视化单张图像
    try:
        cam, text = visualizer.visualize(
            image_path=image_path,
            save_path=args.savedir + 'pic.jpg',
            show=True
        )
        print(f"\nGenerated text: {text}")
    except FileNotFoundError:
        print(f"\nWarning: Image file not found at {image_path}")
        print("Please update the image_path to your test image.")
