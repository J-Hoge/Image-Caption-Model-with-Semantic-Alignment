import re
import matplotlib.pyplot as plt
import numpy as np

# 读取日志文件
log_file_path = '../checkpoints/clip_qf_ve/clip_qf_ve2_trainlog.txt'
output_path = '../checkpoints/clip_qf_ve/clip_qf_ve2.png'
epochs = []
train_losses = []
val_losses = []

with open(log_file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 使用正则表达式提取所有数据
train_pattern = r'Epoch (\d+) - Average training loss: ([\d.]+)'
val_pattern = r'Epoch (\d+) - Average validation loss: ([\d.]+)'

train_matches = re.findall(train_pattern, content)
val_matches = re.findall(val_pattern, content)

# 转换为字典便于查找
val_dict = {int(m[0]): float(m[1]) for m in val_matches}

# 整理训练数据
for match in train_matches:
    epoch = int(match[0])
    train_loss = float(match[1])
    epochs.append(epoch)
    train_losses.append(train_loss)
    if epoch in val_dict:
        val_losses.append(val_dict[epoch])
    else:
        val_losses.append(None)

# 过滤掉没有验证损失的 epoch
valid_indices = [i for i in range(len(epochs)) if val_losses[i] is not None]
epochs_valid = [epochs[i] for i in valid_indices]
train_losses_valid = [train_losses[i] for i in valid_indices]
val_losses_valid = [val_losses[i] for i in valid_indices]

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

# 绘制训练损失曲线（蓝色，圆点标记）
ax.plot(epochs_valid, train_losses_valid, 'b-o', linewidth=2, markersize=6,
        label='Training Loss', color='blue')

# 绘制验证损失曲线（红色，方块标记）
ax.plot(epochs_valid, val_losses_valid, 's-', linewidth=2, markersize=6,
        label='Validation Loss', color='red')

# 设置标题和标签
ax.set_title('Training and Validation Loss Curves (Epochs 1-40)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)

# 设置图例位置
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

# 添加网格
ax.grid(True, linestyle='--', alpha=0.7, color='gray')

# 设置坐标轴范围
ax.set_xlim(min(epochs_valid) - 0.5, max(epochs_valid) + 0.5)
ax.set_ylim(min(min(train_losses_valid), min(val_losses_valid)) - 0.2,
            max(max(train_losses_valid), max(val_losses_valid)) + 0.2)

# 调整布局
plt.tight_layout()

# 保存图片

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已保存到：{output_path}")

# 显示图形
plt.show()

# 打印关键信息
print(f"\n=== 训练统计 ===")
print(f"总 Epoch 数：{len(epochs_valid)}")
print(f"初始训练损失：{train_losses_valid[0]:.4f}")
print(f"最终训练损失：{train_losses_valid[-1]:.4f}")
print(f"初始验证损失：{val_losses_valid[0]:.4f}")
print(f"最终验证损失：{val_losses_valid[-1]:.4f}")
print(f"最佳验证损失：{min(val_losses_valid):.4f} (Epoch {epochs_valid[np.argmin(val_losses_valid)]})")
