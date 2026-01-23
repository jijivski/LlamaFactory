import json
import os
import random

# 设置路径
base_dir = './data' # 根据你的实际目录调整
v2_path = os.path.join(base_dir, 'limo_v2.json')
train_output_path = os.path.join(base_dir, 'limo_v2_train.json')
test_output_path = os.path.join(base_dir, 'limo_v2_test.json')

def split_json_data(input_file, train_count, test_count):
    # 1. 读取原始数据
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 确保数据量足够
    total_needed = train_count + test_count
    if len(data) < total_needed:
        print(f"警告：原始数据只有 {len(data)} 条，不足 {total_needed} 条。")
        # 如果数据不足，可以按比例缩减或取全部
    
    # 2. 随机打乱数据 (可选，但推荐用于训练集制作)
    random.seed(42) # 固定随机种子以便复现
    random.shuffle(data)

    # 3. 切片数据
    train_data = data[:train_count]
    test_data = data[train_count:train_count + test_count]

    # 4. 保存为两个文件
    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
        
    with open(test_output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"成功保存: \n - 训练集: {len(train_data)} 条 -> {train_output_path}\n - 测试集: {len(test_data)} 条 -> {test_output_path}")

# 执行拆分
split_json_data(v2_path, 720, 80)