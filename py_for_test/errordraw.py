import matplotlib.pyplot as plt
import numpy as np
import re
import os

# =================配置区域=================
# 你的三个日志文件路径
log_files = [
    '../checkpoints/clip_qf_ve/clip_qf_ve_trainlog.txt',  # Seed 2026 (假设)
    '../checkpoints/clip_qf_ve/clip_qf_ve2_trainlog.txt',  # Seed 42 (假设)
    '../checkpoints/clip_qf_ve/clip_qf_ve3_trainlog.txt'  # Seed 32 (假设)
]

# 定义颜色，方便区分
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']


# =========================================

def parse_log_file(file_path):
    """解析单个日志文件，返回 epochs, train_losses, val_losses"""
    epochs = []
    train_losses = []
    val_losses = []

    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return None, None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正则表达式匹配
    # 注意：根据你的实际日志格式调整正则。
    # 假设格式为: "Epoch 1 - Average training loss: 4.2074"
    train_pattern = r'Epoch (\d+) - Average training loss: ([\d.]+)'
    val_pattern = r'Epoch (\d+) - Average validation loss: ([\d.]+)'

    train_matches = re.findall(train_pattern, content)
    val_matches = re.findall(val_pattern, content)

    # 转换为数字列表
    for epoch, loss in train_matches:
        epochs.append(int(epoch))
        train_losses.append(float(loss))

    # 验证损失通常和epoch对应，这里假设val_matches的顺序与epochs一致
    # 如果val_loss不是每个epoch都有，需要更复杂的对齐逻辑，这里简化处理
    for i, (epoch, loss) in enumerate(val_matches):
        # 简单对齐：假设val_loss也是从epoch 1开始连续记录的
        # 如果长度不一致，可能需要根据epoch数索引
        pass

        # 为了确保对齐，我们重新构建val_losses列表，基于epochs
    # 创建字典快速查找
    val_dict = {int(epoch): float(loss) for epoch, loss in val_matches}
    aligned_val_losses = [val_dict.get(e, np.nan) for e in epochs]

    return np.array(epochs), np.array(train_losses), np.array(aligned_val_losses)


# =================主程序=================

all_train_losses = []
all_val_losses = []
common_epochs = None

print("Parsing log files...")
for i, file_path in enumerate(log_files):
    epochs, train_l, val_l = parse_log_file(file_path)
    if epochs is not None:
        if common_epochs is None:
            common_epochs = epochs
        else:
            # 确保所有seed的epoch长度一致，取最小公倍数或截断
            min_len = min(len(common_epochs), len(epochs))
            common_epochs = common_epochs[:min_len]
            # 截断已存储的数据
            all_train_losses = [arr[:min_len] for arr in all_train_losses]
            all_val_losses = [arr[:min_len] for arr in all_val_losses]

        all_train_losses.append(train_l[:len(common_epochs)])
        all_val_losses.append(val_l[:len(common_epochs)])
        print(f"Loaded {file_path}: {len(epochs)} epochs")

if not all_train_losses:
    print("No data loaded. Please check file paths and regex patterns.")
else:
    # 转换为 numpy array 以便计算均值和标准差
    # Shape: (num_seeds, num_epochs)
    train_data = np.array(all_train_losses)
    val_data = np.array(all_val_losses)

    # 计算均值和标准差
    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)

    val_mean = np.mean(val_data, axis=0)
    val_std = np.std(val_data, axis=0)

    # =================绘图 1: Loss Curve with Error Bands=================
    plt.figure(figsize=(10, 6))

    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(common_epochs, train_mean, label='Mean Train Loss', color='blue', linewidth=2)
    plt.fill_between(common_epochs, train_mean - train_std, train_mean + train_std,
                     color='blue', alpha=0.2, label='Std Dev')
    plt.title('Training Loss over Epochs (3 Seeds)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 绘制验证损失
    plt.subplot(1, 2, 2)
    plt.plot(common_epochs, val_mean, label='Mean Val Loss', color='red', linewidth=2)
    plt.fill_between(common_epochs, val_mean - val_std, val_mean + val_std,
                     color='red', alpha=0.2, label='Std Dev')
    plt.title('Validation Loss over Epochs (3 Seeds)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('loss_curves_with_error_bands.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =================绘图 2: Final Result Bar Chart with Error Bars=================
    # 获取最后一个epoch的数据作为最终结果
    final_val_mean = val_mean[-1]
    final_val_std = val_std[-1]

    final_train_mean = train_mean[-1]
    final_train_std = train_std[-1]

    metrics = ['Final Train Loss', 'Final Val Loss']
    means = [final_train_mean, final_val_mean]
    stds = [final_train_std, final_val_std]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, means, yerr=stds, capsize=5,
                   color=['skyblue', 'lightcoral'], edgecolor='black', width=0.5)

    # 在柱状图上添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + std + 0.01,
                 f'{mean:.4f} ± {std:.4f}',
                 ha='center', va='bottom', fontsize=10)

    plt.ylabel('Loss Value')
    plt.title('Statistical Stability of EMD Model (Final Results, 3 Seeds)')
    plt.ylim([0, max(means) + max(stds) + 0.1])  # 自动调整Y轴范围
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig('error_bar_final_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nResults Summary:")
    print(f"Final Train Loss: {final_train_mean:.4f} ± {final_train_std:.4f}")
    print(f"Final Val Loss:   {final_val_mean:.4f} ± {final_val_std:.4f}")