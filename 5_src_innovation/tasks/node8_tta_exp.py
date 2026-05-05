import cv2
import numpy as np
import time
import os
import sys

# ==========================================
# 1. 加载测试图像 (优先使用本地已有图片)
# ==========================================
def load_test_image():
    # 候选路径列表：优先搜索你项目里现有的图片
    candidate_paths = [
        'test_image.jpg',
        '1_data/cali_data/detection_test.jpg',
        '../1_data/cali_data/detection_test.jpg',
        '1_data/input_image/bus.jpg' # 备选
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            print(f"✅ 已找到本地测试图像: {path}")
            return cv2.imread(path)
    
    print("❌ 错误: 找不到测试图像！")
    print("由于网络环境限制，自动下载已禁用。")
    print("请手动将一张图片重命名为 'test_image.jpg' 放在当前目录下，或者修改脚本中的路径。")
    sys.exit(1)

# ==========================================
# 2. 核心算法：拉普拉斯方差计算
# ==========================================
def calculate_complexity(image_patch):
    """计算图像块的纹理复杂度 (拉普拉斯方差)"""
    gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

# ==========================================
# 3. 模拟超分推理核心
# ==========================================
def simulate_swinir_1x(patch):
    """模拟单次普通推理 (假设耗时 0.01秒)"""
    time.sleep(0.01) 
    h, w = patch.shape[:2]
    temp = cv2.resize(patch, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_AREA)

def simulate_swinir_8x_tta(patch):
    """模拟 8次 TTA 推理 (假设耗时是 1x 的 8 倍 = 0.08秒)"""
    time.sleep(0.08)
    h, w = patch.shape[:2]
    temp = cv2.resize(patch, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_AREA)

# ==========================================
# 4. 评估指标计算
# ==========================================
def calculate_psnr(img1, img2):
    """计算峰值信噪比 (PSNR)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# ==========================================
# 5. 动态 TTA 调度主管道
# ==========================================
def dynamic_tta_pipeline(image, patch_size=64, threshold=100.0):
    h, w, c = image.shape
    h = h - (h % patch_size)
    w = w - (w % patch_size)
    image = image[:h, :w]
    output_image = np.zeros_like(image)
    total_patches = 0
    tta_count = 0
    
    print(f"\n--- 测试阈值 Threshold = {threshold} ---")
    start_time = time.time()

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            total_patches += 1
            patch = image[i:i+patch_size, j:j+patch_size]
            variance = calculate_complexity(patch)
            
            if variance < threshold:
                sr_patch = simulate_swinir_1x(patch)
            else:
                sr_patch = simulate_swinir_8x_tta(patch)
                tta_count += 1
            output_image[i:i+patch_size, j:j+patch_size] = sr_patch

    end_time = time.time()
    latency = (end_time - start_time) * 1000 
    tta_ratio = (tta_count / total_patches) * 100
    psnr = calculate_psnr(image, output_image)
    
    print(f"  > 耗时: {latency:.2f} ms | PSNR: {psnr:.2f} dB | TTA占比: {tta_ratio:.1f}%")
    return latency, psnr, tta_ratio

# ==========================================
# 6. 执行实验
# ==========================================
def run_experiments():
    image = load_test_image()
    print("图像已加载，尺寸:", image.shape)
    
    # 定义测试阈值 (从全量 TTA 到 全不 TTA)
    thresholds = [0, 200, 800, 2000, 99999]
    results = []
    
    for th in thresholds:
        latency, psnr, ratio = dynamic_tta_pipeline(image, patch_size=128, threshold=th)
        results.append((th, latency, psnr, ratio))
        
    print("\n" + "="*55)
    print("📊 动态 TTA 算力调度实验数据汇总")
    print("="*55)
    print(f"{'阈值(Th)':<10} | {'耗时(ms)':<10} | {'PSNR(dB)':<10} | {'TTA占比(%)':<10}")
    print("-" * 55)
    for res in results:
        th, latency, psnr, ratio = res
        print(f"{th:<10} | {latency:<10.2f} | {psnr:<10.2f} | {ratio:<10.1f}")
    print("="*55)

if __name__ == "__main__":
    run_experiments()