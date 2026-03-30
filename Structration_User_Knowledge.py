# new_Structration_User_Knowledge.py
import os
import re
import json
import jsonlines
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
# 如需 sentence-transformers，可自行切换
# from sentence_transformers import SentenceTransformer

"""
依赖：
  pip install pandas numpy tqdm jsonlines scikit-learn

输入：
  new_batch_output/<dataset>_max<max_his_num>_output.jsonl
  ./data/<dataset>/entity_list.txt
  ./data/<dataset>/kg_final.txt (仅用来估计 relation/不强依赖)
输出：
  ./data/<dataset>/new_user_interest_clustered.txt   # 两列 uid interest
"""

def simple_clean_text(text: str) -> str:
    # 小写 -> 去非字母数字空格 -> 压缩空白
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_llm_jsonl(output_path):
    intents = []      # 所有兴趣短语全集
    uid_phrase = []   # [(uid, phrase), ...]
    lower, upper = 1, 50  # 长度过滤（字符级；你也可改成 token 级）

    with open(output_path, mode="r", encoding="utf-8") as f:
        for ans in jsonlines.Reader(f):
            uid = int(ans["custom_id"])
            raw = ans.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            if not raw:
                continue
            # 约定逗号分割
            cand = [simple_clean_text(x) for x in str(raw).split(",")]
            for it in cand:
                it = it.strip()
                if len(it) > lower and len(it) <= upper:
                    intents.append(it)
                    uid_phrase.append((uid, it))
    return intents, uid_phrase

def encode_intents_tfidf(intents):
    vec = TfidfVectorizer(max_df=0.8, ngram_range=(1,2))
    emb = vec.fit_transform(intents)
    return emb

def main():
    parser = argparse.ArgumentParser(description="Cluster interests and produce new_user_interest_clustered.txt")
    parser.add_argument("--dataset", default="book-crossing", type=str)
    parser.add_argument("--max_his_num", default=30, type=int)
    parser.add_argument("--clusters", default=350, type=int)
    parser.add_argument("--cluster_type", default="tfidf", choices=["tfidf"], type=str)
    args = parser.parse_args()

    dataset = args.dataset
    data_path = f"./data/{dataset}"
    out_interest = os.path.join(data_path, "new_user_interest_clustered.txt")

    in_jsonl = os.path.join("new_batch_output", f"{dataset}_max{args.max_his_num}_output.jsonl")
    assert os.path.exists(in_jsonl), f"not found: {in_jsonl}"

    # 读实体列表，计算 entity_max
    entity_list = []
    with open(os.path.join(data_path,"entity_list.txt"), "r", encoding="utf-8") as f:
        for idx, l in enumerate(f):
            if idx == 0:  # 跳过header
                continue
            l = l.strip()
            if not l:
                continue
            sp = l.split()
            if len(sp) >= 2:
                entity_list.append(int(sp[1]))
    entity_max = max(entity_list) + 1 if entity_list else 0

    # 读取 LLM 产出的兴趣短语
    intents, uid_phrase = load_llm_jsonl(in_jsonl)
    print(f"[INFO] collected phrases: {len(intents)} (unique {len(set(intents))})")

    if not intents:
        # 写空文件以防止后续流程崩
        pd.DataFrame(columns=["uid","interest"]).to_csv(out_interest, sep=" ", index=False, header=False)
        print(f"[WARN] no phrases found, wrote empty file: {out_interest}")
        return

    # 向量化
    if args.cluster_type == "tfidf":
        emb = encode_intents_tfidf(intents)
    else:
        raise NotImplementedError

    # 聚类
    C = min(args.clusters, emb.shape[0])  # 防止簇数 > 样本数
    print(f"[INFO] KMeans clusters: {C}")
    kmeans = KMeans(n_clusters=C, n_init="auto", random_state=42)
    cluster_ids = kmeans.fit_predict(emb)

    # 映射 短语->簇id
    phrase2cid = {}
    for phrase, cid in zip(intents, cluster_ids):
        phrase2cid[phrase] = cid

    # 给每个簇分配“兴趣实体ID段”
    phrase2eid = {p: (c + entity_max) for p, c in phrase2cid.items()}

    # 生成 uid -> merged_interest 边
    rows = []
    for uid, phrase in uid_phrase:
        if phrase in phrase2eid:
            rows.append([uid, phrase2eid[phrase]])
    df = pd.DataFrame(rows, columns=["uid","merged_interest"])

    # 基于簇规模过滤极端簇
    user_num = df["uid"].nunique()
    grp = df.groupby("merged_interest")["uid"].count().sort_values(ascending=False)
    del_list = set(grp[grp >= int(user_num/5)].index.tolist())  # 过大簇
    del_list |= set(grp[grp == 1].index.tolist())               # 单例簇
    before = len(df)
    df = df[~df["merged_interest"].isin(del_list)]
    after = len(df)
    print(f"[INFO] filtered edges: {before} -> {after}")
    print(f"[INFO] sparsity ~ {round(after/(max(user_num,1)*C),4)*100}%")

    # 导出为两列 uid interest
    out_df = df[["uid","merged_interest"]].rename(columns={"uid":"uid","merged_interest":"interest"})
    out_df.to_csv(out_interest, sep=" ", index=False, header=False)
    print(f"[OK] wrote: {out_interest}  (rows={len(out_df)})")

if __name__ == "__main__":
    main()
