import os
import json
from pathlib import Path

# 设置你的路径
dataset_path = Path("experiments/revoarm_bottle/offline_dataset/revoarm_bottle_success")
meta_path = dataset_path / "meta/episodes.jsonl"
data_path = dataset_path / "data"

print(f"检查路径: {dataset_path}")

# 1. 读取账本
if not meta_path.exists():
    print("❌ 错误: 找不到 meta/episodes.jsonl")
    exit()

with open(meta_path, 'r') as f:
    episodes = [json.loads(line) for line in f]

print(f"📖 账本里共有 {len(episodes)} 个 episode")

# 2. 检查货物
missing_count = 0
for i, ep in enumerate(episodes):
    # LeRobot V2 通常按 chunk 存储，但也可能不按。
    # 我们假设文件名是 episode_{id}.parquet
    ep_id = ep.get("episode_index", i)
    chunk_id = ep_id // 1000 # 默认 1000 个一包，或者是其他逻辑
    
    # 尝试几种可能的路径
    possible_paths = [
        data_path / f"episode_{ep_id:06d}.parquet",
        data_path / f"chunk-{chunk_id:03d}/episode_{ep_id:06d}.parquet"
    ]
    
    found = False
    for p in possible_paths:
        if p.exists():
            found = True
            break
            
    if not found:
        print(f"❌ 缺货: Episode {ep_id} 找不到文件!")
        print(f"   尝试寻找: {possible_paths}")
        missing_count += 1
        if missing_count > 5:
            print("... 缺失太多，停止打印 ...")
            break

if missing_count == 0:
    print("✅ 完美！所有文件都在。那可能是其他 Assertion 问题。")
else:
    print(f"🚫 总共缺失 {missing_count} 个文件。请修改 episodes.jsonl 删掉这些行。")