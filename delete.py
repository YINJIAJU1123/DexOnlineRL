import os
import shutil
import json
import math

# ================= 配置区域 =================
# 1. 源数据路径 (Chunk 结构的原始数据)
SRC_ROOT = "/home/yin/Online_RL/HIL-RL/experiments/revoarm_bottle/offline_dataset/revoarm_bottle2"

# 2. 目标数据路径 (生成的 Mini 数据集)
DST_ROOT = "/home/yin/Online_RL/HIL-RL/experiments/revoarm_bottle/offline_dataset/revoarm_bottle"

# 3. 保留前 5 条
KEEP_EPISODES = 1
# ===========================================

def make_mini_v2():
    print(f"🚀 开始制作 Mini 数据集 (从 Chunk 结构提取前 {KEEP_EPISODES} 条)...")
    
    # --- 0. 检查源路径 ---
    # 既然你说在 chunk-000 下，我们直接去那里找
    src_data_chunk = os.path.join(SRC_ROOT, "data", "chunk-000")
    if not os.path.exists(src_data_chunk):
        print(f"❌ 错误：找不到源数据目录: {src_data_chunk}")
        print("请确认你的源数据是否真的有 chunk-000 文件夹")
        return

    # --- 1. 清理旧目标 ---
    if os.path.exists(DST_ROOT):
        shutil.rmtree(DST_ROOT)
    os.makedirs(DST_ROOT)

    # --- 2. 复制 Parquet 数据 (Flatten: 从 chunk 里拿出来) ---
    print("📦 正在提取 Parquet 文件...")
    dst_data = os.path.join(DST_ROOT, "data")
    os.makedirs(dst_data)

    # 找到 chunk-000 里的所有 parquet
    all_files = sorted([f for f in os.listdir(src_data_chunk) if f.endswith(".parquet")])
    if len(all_files) == 0:
        print("❌ 错误：chunk-000 里没有 .parquet 文件！")
        return
    
    # 只取前 5 个
    files_to_copy = all_files[:KEEP_EPISODES]
    
    for fname in files_to_copy:
        src_file = os.path.join(src_data_chunk, fname)
        dst_file = os.path.join(dst_data, fname) # 注意：目标不再有 chunk-000，直接放在 data 下
        shutil.copy(src_file, dst_file)
    
    print(f"✅ 已复制 {len(files_to_copy)} 个 Parquet 文件 (已扁平化)")

    # --- 3. 复制视频文件 (如果有) ---
    # 源结构: videos/chunk-000/camera_name/episode_xxx.mp4
    # 目标结构: videos/camera_name/episode_xxx.mp4 (扁平化)
    src_video_chunk = os.path.join(SRC_ROOT, "videos", "chunk-000")
    if os.path.exists(src_video_chunk):
        print("🎥 正在提取视频文件...")
        dst_video_root = os.path.join(DST_ROOT, "videos")
        os.makedirs(dst_video_root)

        # 遍历所有相机文件夹 (如 observation.images.cam_chest)
        camera_dirs = [d for d in os.listdir(src_video_chunk) if os.path.isdir(os.path.join(src_video_chunk, d))]
        
        for cam_name in camera_dirs:
            src_cam_dir = os.path.join(src_video_chunk, cam_name)
            dst_cam_dir = os.path.join(dst_video_root, cam_name)
            os.makedirs(dst_cam_dir)
            
            # 同样只复制前 5 个视频
            video_files = sorted([f for f in os.listdir(src_cam_dir) if f.endswith(".mp4")])
            for vname in video_files[:KEEP_EPISODES]:
                shutil.copy(os.path.join(src_cam_dir, vname), os.path.join(dst_cam_dir, vname))
            print(f"   -> 相机 {cam_name}: 复制完成")

    # --- 4. 处理 Meta 元数据 ---
    print("📊 处理元数据...")
    src_meta = os.path.join(SRC_ROOT, "meta")
    dst_meta = os.path.join(DST_ROOT, "meta")
    os.makedirs(dst_meta)

    # 4.1 复制 stats.json (直接复制)
    if os.path.exists(os.path.join(src_meta, "stats.json")):
        shutil.copy(os.path.join(src_meta, "stats.json"), os.path.join(dst_meta, "stats.json"))
    
    # 4.2 复制 tasks.jsonl (如果有，直接复制)
    if os.path.exists(os.path.join(src_meta, "tasks.jsonl")):
         shutil.copy(os.path.join(src_meta, "tasks.jsonl"), os.path.join(dst_meta, "tasks.jsonl"))

    # 4.3 修改 info.json
    if os.path.exists(os.path.join(src_meta, "info.json")):
        with open(os.path.join(src_meta, "info.json"), 'r') as f:
            info = json.load(f)
        
        # 重新计算帧数
        orig_frames = info.get("total_frames", 0)
        total_src_files = len(all_files)
        new_frames = math.ceil((orig_frames / total_src_files) * KEEP_EPISODES)
        
        info["total_episodes"] = KEEP_EPISODES
        info["total_frames"] = new_frames
        
        # 关键：删除 chunks 定义，因为我们把数据扁平化了
        if "chunks" in info:
            del info["chunks"]

        with open(os.path.join(dst_meta, "info.json"), 'w') as f:
            json.dump(info, f, indent=4)
        print("✅ info.json 已修正 (Removed chunks definition)")

    print("-" * 30)
    print(f"🎉 Mini 数据集制作完成！")
    print(f"📂 位置: {DST_ROOT}")
    print("提示：该数据集已去除 chunk-000 文件夹，结构扁平，适合 Learner 读取。")

if __name__ == "__main__":
    make_mini_v2()