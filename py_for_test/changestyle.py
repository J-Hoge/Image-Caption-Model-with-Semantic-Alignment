import json


def fix_coco_submission(input_file, output_file):
    """
    将错误的提交格式转换为 COCO 官方格式
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fixed_data = []
    for item in data:
        # 转换 image_id 为整数（去掉前导零）
        image_id = int(item['image_id'])

        # 只保留必需字段
        fixed_item = {
            'image_id': image_id,
            'caption': item['caption']
        }
        fixed_data.append(fixed_item)

    # 保存为正确格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 已转换 {len(fixed_data)} 条结果")
    print(f"📁 输出文件：{output_file}")

    # 验证前 3 条
    print("\n📋 前 3 条示例:")
    for item in fixed_data[:3]:
        print(f"  {item}")

    return fixed_data


# 使用示例
if __name__ == "__main__":
    fix_coco_submission(
        input_file='../data/test_out/best_test.json',
        output_file='../data/test_out/best_test_submit.json'
    )