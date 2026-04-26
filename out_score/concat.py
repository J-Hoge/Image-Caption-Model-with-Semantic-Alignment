from typing import List
from PIL import Image
from matplotlib import pyplot as plt


def combine_images_horizontally(image_paths: List[str], save_path: str, titles: List[str] = None, figsize=(25, 6)):
    """
    将多张图片横向拼接

    Args:
        image_paths: 图片路径列表
        save_path: 保存路径
        titles: 每张图的标题列表（可选）
        figsize: 整体图形大小
    """
    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=figsize)

    if num_images == 1:
        axes = [axes]

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        axes[i].imshow(img)
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout(w_pad=1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"拼接图已保存到：{save_path}")
    plt.close()


image_paths = [
    'bestpic_original.jpg',
    'basepic_heatmap.jpg',
    'basepic_cam.jpg',
    'clippic_heatmap.jpg',
    'clippic_cam.jpg',
    'bestpic_heatmap.jpg',
    'bestpic_cam.jpg'
]
titles = ['Original', 'BASE Heatmap', 'BASE CAM', 'E Heatmap', 'E CAM', 'EMD Heatmap', 'EMD CAM']
combine_images_horizontally(image_paths, 'EEMD4.jpg', titles)
