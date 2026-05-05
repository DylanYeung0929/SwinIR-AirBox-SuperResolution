import cv2
import numpy as np
import os
import time

class DegradationEstimator:
    def __init__(self):
        print("[Node 14] 启动盲超分退化感知引擎 (Blind SR Perception)...")

    def estimate_noise_level(self, gray_img):
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        return np.mean(np.abs(laplacian))

    def estimate_jpeg_blocking(self, gray_img):
        h, w = gray_img.shape
        diff_h = np.abs(gray_img[:, 7:-1:8].astype(int) - gray_img[:, 8::8].astype(int))
        diff_v = np.abs(gray_img[7:-1:8, :].astype(int) - gray_img[8::8, :].astype(int))
        block_score = (np.mean(diff_h) + np.mean(diff_v)) / 2.0
        return block_score

    def analyze_patch(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        noise_score = self.estimate_noise_level(gray)
        block_score = self.estimate_jpeg_blocking(gray)
        degradation_idx = (noise_score * 0.4) + (block_score * 2.5)
        return noise_score, block_score, degradation_idx

def simulate_swinir_naive(img):
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(upscaled, -1, kernel)

def run_blind_sr_experiment():
    test_img_path = "../1_data/cali_data/detection_test.jpg"
    if not os.path.exists(test_img_path):
        test_img_path = "1_data/cali_data/detection_test.jpg"
        if not os.path.exists(test_img_path):
            print("找不到测试图像，生成模拟图像...")
            clean_img = np.ones((400, 400, 3), dtype=np.uint8) * 200
            cv2.putText(clean_img, "TEST", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 50, 50), 5)
        else:
            clean_img = cv2.imread(test_img_path)
    else:
        clean_img = cv2.imread(test_img_path)

    h, w = clean_img.shape[:2]
    center_y, center_x = h // 2, w // 2
    crop_size = 128
    clean_patch = clean_img[center_y-crop_size:center_y+crop_size, center_x-crop_size:center_x+crop_size]

    # 1. 模拟退化
    noise = np.random.normal(0, 15, clean_patch.shape).astype(np.float32)
    noisy_patch = np.clip(clean_patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
    _, encimg = cv2.imencode('.jpg', noisy_patch, encode_param)
    degraded_patch = cv2.imdecode(encimg, 1)

    # 2. 感知退化
    estimator = DegradationEstimator()
    start_time = time.time()
    n_score, b_score, deg_idx = estimator.analyze_patch(degraded_patch)
    cost_ms = (time.time() - start_time) * 1000

    # 3. 路由对抗策略
    naive_sr_out = simulate_swinir_naive(degraded_patch)

    if deg_idx > 30.0:
        cleaned_patch = cv2.bilateralFilter(degraded_patch, d=5, sigmaColor=50, sigmaSpace=50)
        smart_sr_out = simulate_swinir_naive(cleaned_patch)
    else:
        smart_sr_out = simulate_swinir_naive(degraded_patch)

    # 4. 生成带数据面板的可视化大图
    vis_degraded = cv2.resize(degraded_patch, (smart_sr_out.shape[1], smart_sr_out.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    cv2.putText(vis_degraded, "1. Input (Degraded)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(naive_sr_out, "2. Naive SR (Artifacts)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(smart_sr_out, "3. Our Blind SR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    panel_y = 65
    cv2.putText(vis_degraded, f"Degradation Index: {deg_idx:.2f}", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(vis_degraded, f"Noise: {n_score:.1f} | Block: {b_score:.1f}", (10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(vis_degraded, f"Perception Cost: {cost_ms:.3f} ms", (10, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    if deg_idx > 30.0:
        cv2.putText(smart_sr_out, "Action: Triggered Pre-Denoise", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    final_showcase = np.hstack((vis_degraded, naive_sr_out, smart_sr_out))
    
    save_dir = "/home/dylanyeung/SwinIR_AirBox_Project/5_src_innovation/results/node14"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "blind_sr_comparison.jpg")
    cv2.imwrite(save_path, final_showcase)
    
    print("\n" + "="*65)
    print("🏆 盲超分防御架构验证战报 (Blind SR Report)")
    print("="*65)
    print(f"🎯 噪点浓度 (Noise Level)   : {n_score:.2f}")
    print(f"🎯 马赛克指数 (Blockiness) : {b_score:.2f}")
    print(f"🚨 综合退化指数 (Deg Index) : {deg_idx:.2f} (安全阈值: 30.0)")
    print(f"⚡ 感知大脑耗时 (Overhead)  : {cost_ms:.3f} ms")
    
    action = "✅ 触发前置保边清洗 (Triggered Pre-Denoise)" if deg_idx > 30.0 else "➖ 图像干净，直接送入 TPU (Direct SR)"
    print(f"🛡️ 动态防御执行动作         : {action}")
    print("-" * 65)
    print(f"📸 包含量化 OSD 数据面板的对比大图已保存至:\n   {save_path}")
    print("="*65)

if __name__ == "__main__":
    run_blind_sr_experiment()
