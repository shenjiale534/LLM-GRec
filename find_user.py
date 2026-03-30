import json
import argparse
import numpy as np
import pandas as pd
from utils import MyLoader  # <--- 这里修正了，用你代码里的真实类名

def find_sparse_user():
    # 1. 设置配置 (强制指定为 book-crossing，因为你要做这个数据集的案例)
    dataset_name = 'book-crossing' 
    print(f"正在读取 {dataset_name} 的配置文件...")
    
    try:
        config = json.load(open(f'./config/{dataset_name}.json'))
    except FileNotFoundError:
        print(f"错误：找不到 ./config/{dataset_name}.json 文件。请检查文件名是否正确 (例如是否叫 book-crossing.json 或 book_crossing.json)")
        return

    # 设置设备等基本参数 (防止 Loader 报错)
    config['device'] = 'cpu' # 找用户不需要 GPU
    config['dataset'] = dataset_name
    
    # 2. 初始化加载器
    print("正在加载数据 (MyLoader)...")
    loader = MyLoader(config)
    
    # 3. 获取训练集和测试集数据
    # 根据你的 train.py，loader.train 是一个 DataFrame
    train_df = loader.train
    
    # 尝试获取测试集字典 (通常是 testDict 或 test_dict)
    if hasattr(loader, 'testDict'):
        test_dict = loader.testDict
    elif hasattr(loader, 'test_dict'):
        test_dict = loader.test_dict
    else:
        # 如果都没有，尝试手动从 loader.test (如果是DF) 构建
        print("正在从 loader.test 构建测试集字典...")
        try:
            test_dict = loader.test.groupby('userid')['itemid'].apply(list).to_dict()
        except:
            print("无法找到测试集数据，请检查 MyLoader 的属性。")
            return

    print("-" * 50)
    print("开始筛选稀疏用户 (历史交互 2-5 本)...")
    print("-" * 50)

    # 4. 统计每个用户的历史长度
    # train_df 应该有 'userid' 和 'itemid' 列
    user_interact_counts = train_df.groupby('userid')['itemid'].apply(list)
    
    candidates_found = 0
    
    # 遍历所有用户
    for uid, history_items in user_interact_counts.items():
        history_len = len(history_items)
        
        # === 筛选条件 ===
        if 2 <= history_len <= 5:
            # 检查这个用户在不在测试集里 (有没有要预测的 Truth)
            if uid in test_dict:
                target_items = test_dict[uid]
                
                # 打印符合条件的用户
                print(f"【候选用户 ID】: {uid}")
                print(f"  - 历史交互长度: {history_len}")
                print(f"  - 历史物品 IDs: {history_items}") 
                print(f"  - 测试目标 IDs (Ground Truth): {target_items}")
                print("-" * 30)
                
                candidates_found += 1
                
                # 只要前 5 个就够了
                if candidates_found >= 5:
                    break
    
    if candidates_found == 0:
        print("未找到符合条件的用户，可能是筛选条件太严，或者数据集中没有极度稀疏的用户。")

if __name__ == "__main__":
    find_sparse_user()