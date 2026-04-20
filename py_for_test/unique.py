import json

def remove_duplicate_image_ids(input_file, output_file):
    """
    去除 JSON 文件中重复的 image_id，只保留第一次出现的记录

    Args:
        input_file: 输入 JSON 文件路径
        output_file: 输出 JSON 文件路径
    """
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 用于存储去重后的结果
    deduplicated_data = []
    # 用于记录已经出现过的 image_id
    seen_image_ids = set()

    # 遍历每条数据
    for item in data:
        image_id = item['image_id']
        # 如果这个 image_id 还没出现过，就添加到结果中
        if image_id not in seen_image_ids:
            deduplicated_data.append(item)
            seen_image_ids.add(image_id)

    # 将去重后的数据写入新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduplicated_data, f, indent=2, ensure_ascii=False)

    print(f"原始数据条数：{len(data)}")
    print(f"去重后数据条数：{len(deduplicated_data)}")
    print(f"删除重复条数：{len(data) - len(deduplicated_data)}")
    print(f"结果已保存到：{output_file}")

# 使用示例
if __name__ == "__main__":
    input_file = r'D:\python_code\clipcap\data\val_out\clip_qf_ve_val.json'
    output_file = r'D:\python_code\clipcap\data\val_out\captions_val2017_jhoge_results.json'

    remove_duplicate_image_ids(input_file, output_file)
