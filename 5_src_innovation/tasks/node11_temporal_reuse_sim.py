import cv2
import numpy as np
import time
import os

def calculate_frame_diff(patch_t, patch_t1):
    """计算前后两帧 patch 的均方误差 (MSE)"""
    return np.mean((patch_t.astype(float) - patch_t1.astype(float)) ** 2)

def simulate_temporal_reuse(image_path, threshold=5.0):
    """
    模拟时序复用过程
    threshold: 差异阈值。低于此值则复用旧块。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 找不到图像 {image_path}，已自动生成 1080P 模拟帧测试...")
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    h, w, _ = img.shape
    patch_size = 128
    
    # 模拟两帧：frame_t1 是原图，frame_t 是加入局部动态后的图
    frame_t1 = img[:h - (h%patch_size), :w - (w%patch_size)]
    # 模拟下一帧：80% 区域不动，20% 区域有动态（引入模糊/噪声模拟运动）
    frame_t = frame_t1.copy()
    noise_mask = np.random.rand(frame_t.shape[0]//patch_size, frame_t.shape[1]//patch_size) > 0.8
    
    for i in range(noise_mask.shape[0]):
        for j in range(noise_mask.shape[1]):
            if noise_mask[i,j]:
                # 模拟物体移动或光影变化
                y, x = i*patch_size, j*patch_size
                frame_t[y:y+patch_size, x:x+patch_size] = cv2.GaussianBlur(frame_t[y:y+patch_size, x:x+patch_size], (15,15), 0)

    print(f"\n==========================================")
    print(f"🎬 节点 11: 视频流时序稀疏复用验证 (Temporal Sparse Reuse)")
    print(f"==========================================")
    
    total_patches = 0
    reuse_count = 0
    start_time = time.time()

    for i in range(0, frame_t.shape[0], patch_size):
        for j in range(0, frame_t.shape[1], patch_size):
            total_patches += 1
            patch_now = frame_t[i:i+patch_size, j:j+patch_size]
            patch_prev = frame_t1[i:i+patch_size, j:j+patch_size]
            
            diff = calculate_frame_diff(patch_now, patch_prev)
            
            if diff < threshold:
                # 【核心创新】: 背景没动，直接白嫖显存里的上一帧！模拟极速复用
                time.sleep(0.0001)
                reuse_count += 1
            else:
                # 【正常逻辑】: 有运动目标，老老实实跑 TPU，模拟耗时
                time.sleep(0.01)

    end_time = time.time()
    latency = (end_time - start_time) * 1000
    
    reuse_rate = (reuse_count / total_patches) * 100
    fps_boost_factor = 100 / (100 - reuse_rate)
    
    print(f"✅ 处理完成！总分块数: {total_patches}")
    print(f"📦 静态缓存复用率 (Cache Hit): {reuse_rate:.2f}%")
    print(f"⏱️ 综合推理耗时: {latency:.2f} ms")
    print(f"🚀 模拟 FPS 收益倍率: {fps_boost_factor:.2f} 倍 (强行拔高帧率！)")
    print(f"==========================================\n")
    
    return reuse_rate, fps_boost_factor

if __name__ == "__main__":
    # 使用你目录里的测试图
    simulate_temporal_reuse("1_data/input_image/detection_test.jpg")
