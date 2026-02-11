import pandas as pd
import numpy as np
import json
import os
import shutil
import glob

# ================= 配置区 =================
# 1. 你的原始数据集路径 (26维)
INPUT_DIR = "/home/lixin/OnlineRl/experiments/revoarm_bottle/offline_dataset/revoarm_bottle_23"  # 这里填你原本26维的那个文件夹

# 2. 新数据集输出路径 (23维)
OUTPUT_DIR = "/home/lixin/OnlineRl/experiments/revoarm_bottle/offline_dataset/revoarm_bottle_cleaned_23dim"

# 3. 目标维度
TARGET_DIM = 23
# ==========================================

def clean_dataset():
    if os.path.exists(OUTPUT_DIR):
        print(f"⚠️ 输出目录已存在，建议删除或改名: {OUTPUT_DIR}")
        return
    os.makedirs(OUTPUT_DIR)
    
    print(f"🚀 开始数据清洗: 26维 -> {TARGET_DIM}维")

    # === 1. 复制 meta 文件夹 (除了我们要改的文件) ===
    src_meta = os.path.join(INPUT_DIR, "meta")
    dst_meta = os.path.join(OUTPUT_DIR, "meta")
    os.makedirs(dst_meta, exist_ok=True)
    
    # === 2. 清洗 info.json ===
    print("Processing info.json...")
    with open(os.path.join(src_meta, "info.json"), 'r') as f:
        info = json.load(f)
    
    # 修改 shape 和 names
    for key in ['observation.state', 'action']:
        if key in info['features']:
            info['features'][key]['shape'] = [TARGET_DIM]
            # 只保留前 23 个名字
            old_names = info['features'][key]['names']
            if old_names and len(old_names) >= TARGET_DIM:
                info['features'][key]['names'] = old_names[:TARGET_DIM]
    
    with open(os.path.join(dst_meta, "info.json"), 'w') as f:
        json.dump(info, f, indent=4)

    # === 3. 清洗 stats.json ===
    print("Processing stats.json...")
    with open(os.path.join(src_meta, "stats.json"), 'r') as f:
        stats = json.load(f)

    # 递归处理所有列表，截断到 23
    def recursive_slice(d):
        for k, v in d.items():
            if isinstance(v, dict):
                recursive_slice(v)
            elif isinstance(v, list):
                # 只有长度 >= 26 的列表才切，防止切错（比如 timestamp 这种长度为1的）
                if len(v) >= 26: 
                    d[k] = v[:TARGET_DIM]

    recursive_slice(stats)
    
    with open(os.path.join(dst_meta, "stats.json"), 'w') as f:
        json.dump(stats, f, indent=4)
        
    # === 4. 复制 videos 文件夹 (直接硬链接或复制) ===
    print("Linking videos...")
    src_videos = os.path.join(INPUT_DIR, "videos")
    dst_videos = os.path.join(OUTPUT_DIR, "videos")
    if os.path.exists(src_videos):
        shutil.copytree(src_videos, dst_videos) # 视频不用改

    # === 5. 核心：清洗 Parquet 数据 ===
    print("Processing Parquet files...")
    data_files = glob.glob(os.path.join(INPUT_DIR, "data/chunk-*/episode_*.parquet"))
    
    for file_path in data_files:
        # 构建输出路径
        rel_path = os.path.relpath(file_path, INPUT_DIR)
        out_path = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # 读取
        df = pd.read_parquet(file_path)
        
        # 处理 State
        if 'observation.state' in df.columns:
            # 每一行都是一个 numpy array 或 list，我们需要切片
            df['observation.state'] = df['observation.state'].apply(
                lambda x: x[:TARGET_DIM] if len(x) >= TARGET_DIM else x
            )
            
        # 处理 Action
        if 'action' in df.columns:
            df['action'] = df['action'].apply(
                lambda x: x[:TARGET_DIM] if len(x) >= TARGET_DIM else x
            )
            
        # 保存
        df.to_parquet(out_path)
        print(f"-> Saved: {rel_path}")

    # === 6. 复制 tasks.jsonl ===
    shutil.copy(os.path.join(src_meta, "tasks.jsonl"), os.path.join(dst_meta, "tasks.jsonl"))

    print("\n✅ 数据清洗完成！")
    print(f"新的数据集路径: {OUTPUT_DIR}")
    print("请修改 config，将 dataset_repo_id 指向这个新路径。")

if __name__ == "__main__":
    # 需要先安装依赖: pip install pandas pyarrow
    clean_dataset()