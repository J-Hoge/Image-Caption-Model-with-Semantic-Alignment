import matplotlib.pyplot as plt
import re
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_log_file(file_path):
    """解析训练日志文件，提取 epoch、train loss 和 val loss"""
    epochs = []
    train_losses = []
    val_losses = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配训练损失行
            train_match = re.search(r'Epoch (\d+) - Average training loss: ([\d.]+)', line)
            if train_match:
                epoch = int(train_match.group(1))
                train_loss = float(train_match.group(2))
                epochs.append(epoch)
                train_losses.append(train_loss)

            # 匹配验证损失行
            val_match = re.search(r'Epoch (\d+) - Average validation loss: ([\d.]+)', line)
            if val_match:
                val_loss = float(val_match.group(2))
                val_losses.append(val_loss)

    return epochs, train_losses, val_losses

def plot_training_curves():
    """绘制 8 个模型的训练曲线"""

    # 定义 8 个日志文件的路径和标签
    log_files = [
        ('checkpoints/base/base_trainlog.txt', 'Base'),
        # ('checkpoints/clip/clip_trainlog.txt', 'E'),
        # ('checkpoints/clip_qf/clip_qf_trainlog.txt', 'EM'),
        # ('checkpoints/clip_ve/clip_ve_trainlog.txt', 'ED'),
        # ('checkpoints/qf/qf_trainlog.txt', 'M'),
        ('checkpoints/clip_qf_ve/clip_qf_ve_trainlog.txt', 'EMD'),
        ('checkpoints/qf_ve/qf_ve_trainlog.txt', 'MD'),
        # ('checkpoints/ve/ve_trainlog.txt', 'D'),
    ]

    # 定义 8 种颜色
    colors = [
        '#FF6B6B',  # 红色
        '#F7DC6F',  # 黄色
        '#BB8FCE',  # 紫色
        '#4ECDC4',  # 青色
        '#45B7D1',  # 蓝色
        '#FFA07A',  # 橙色
        '#98D8C8',  # 绿色
        '#85C1E9',  # 浅蓝
    ]

    # 创建图形，设置较大的尺寸
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)

    # 遍历每个文件并绘制曲线
    for idx, (file_path, label) in enumerate(log_files):
        full_path = Path('D:/python_code/clipcap') / file_path

        if not full_path.exists():
            print(f"警告：文件不存在 - {full_path}")
            continue

        epochs, train_losses, val_losses = parse_log_file(full_path)

        color = colors[idx % len(colors)]

        # 绘制训练损失（实线）
        ax.plot(epochs, train_losses, linestyle='-', color=color,
                linewidth=2, label=f'{label} Train', alpha=0.8)

        # 绘制验证损失（虚线）
        ax.plot(epochs, val_losses, linestyle='-.', color=color,
                linewidth=2, label=f'{label} Val', alpha=0.8)

    # 设置标题和标签
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training and Validation Loss Curves for 8 Models',
                 fontsize=16, fontweight='bold', pad=20)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    ax.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.9)

    # 设置 y 轴为对数刻度（可选，如果 loss 变化范围大）
    # ax.set_yscale('log')

    # 调整布局
    plt.tight_layout()

    # 保存图形
    output_path = Path('D:/python_code/clipcap/out_score/all.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图形已保存到：{output_path}")

    # 显示图形
    plt.show()

if __name__ == '__main__':
    plot_training_curves()
