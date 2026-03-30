import os

def load_id_map():
    # 这里假设你的映射文件叫 item_list.txt
    # 格式通常是: org_id(ISBN)  remap_id
    # 或者: remap_id  org_id
    path = './data/book-crossing/item_list.txt'
    
    id2name = {}
    
    if not os.path.exists(path):
        print(f"找不到文件: {path}")
        print("请检查 data/book-crossing 下有没有 item_list.txt 或 entity_list.txt")
        return None

    print(f"正在读取 {path} ...")
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        # 跳过第一行 header (如果有)
        lines = f.readlines()
        for line in lines[1:]: 
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            # 这里需要你根据文件实际内容调整
            # 假设第一列是原始ID(ISBN/书名)，第二列是数字ID
            # 如果你的文件反过来，请把下面两行互换
            org_id = parts[0] 
            remap_id = parts[1]
            
            try:
                id2name[int(remap_id)] = org_id
            except ValueError:
                continue
    return id2name

def main():
    # 刚才筛选出来的候选用户数据
    candidates = [
        {'uid': 0, 'history': [7347, 7348, 7349, 7351], 'target': [7350]},
        {'uid': 2, 'history': [6720, 7361, 7363, 7364], 'target': [7362]},
        {'uid': 6, 'history': [7396, 7397, 7398, 7399], 'target': [7395]},
        {'uid': 8, 'history': [7368, 7468, 7470, 7471], 'target': [7469]},
        {'uid': 9, 'history': [7472, 7473, 7474, 7475], 'target': [7476, 7477]},
    ]
    
    id2name = load_id_map()
    if not id2name:
        return

    print("-" * 60)
    for user in candidates:
        uid = user['uid']
        print(f"User ID: {uid}")
        
        # 翻译历史记录
        history_names = [id2name.get(i, str(i)) for i in user['history']]
        print(f"History: {history_names}")
        
        # 翻译目标
        target_names = [id2name.get(i, str(i)) for i in user['target']]
        print(f"Target : {target_names}")
        print("-" * 60)

if __name__ == "__main__":
    main()