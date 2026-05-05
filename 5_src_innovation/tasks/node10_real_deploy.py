import cv2
import numpy as np
import sophon.sail as sail
import os
import time
import sys
import argparse

# 确保能找到 core 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.inference import SwinIRInference

class EdgeCloudReceiver:
    def __init__(self, bmodel_path, device_id=0):
        """初始化边缘端接收器 (AirBox 端)"""
        print(f"🚀 [AirBox 边缘端] 正在挂载 TPU 引擎: {os.path.basename(bmodel_path)}")
        self.handle = sail.Handle(device_id)
        self.bmcv = sail.Bmcv(self.handle)
        
        if not os.path.exists(bmodel_path):
            raise FileNotFoundError(f"❌ 找不到 bmodel 模型: {bmodel_path}")
            
        self.engine = sail.Engine(bmodel_path, device_id, sail.IOMode.SYSIO)
        self.swinir = SwinIRInference(self.engine)
        
    def receive_and_reconstruct(self, compressed_stream, args):
        """接收云端传来的压缩字节流并进行 TPU 实机超分"""
        
        # 1. 硬件解码 (将低带宽流解码为 BMImage)
        # 实际部署中，这里应该用 VPU 进行 H.264 解码，此处用 cv2 模拟解码过程接收报文
        decode_start = time.time()
        dec_img = cv2.imdecode(np.frombuffer(compressed_stream, np.uint8), 1)
        decode_time = (time.time() - decode_start) * 1000
        
        # 2. 转换为硬件格式 (Zero-Copy 准备)
        img_bm = sail.BMImage()
        self.bmcv.mat_to_bm_image(dec_img, img_bm)
        
        # 3. 调用真实的 SwinIR bmodel 进行 4x 提升
        print(f"⚡ [AirBox 边缘端] 接收到低分流，启动 TPU 硬件重构...")
        tpu_start = time.time()
        
        # 实际调用我们节点 0-6 优化的推理逻辑
        sr_img = self.swinir.process_image(img_bm, self.bmcv, args, None)
        
        tpu_time = (time.time() - tpu_start) * 1000
        print(f"✅ [AirBox 边缘端] 重建完成! 解码耗时: {decode_time:.2f}ms | TPU推理耗时: {tpu_time:.2f}ms")
        
        return dec_img, sr_img

def run_real_integration_test(args):
    print("\n" + "="*55)
    print("🌐 [系统联调] 节点 10: 端云协同带宽节省与硬件实机重建")
    print("="*55)

    if not os.path.exists(args.img_path):
        print(f"❌ 找不到测试源图像: {args.img_path}")
        return

    # 1. 【模拟云端】把 4K 压缩成 480P 字节流
    print("☁️ [云端] 正在对高分辨率源视频进行极致压缩 (模拟弱网传输)...")
    raw_img = cv2.imread(args.img_path)
    org_size = os.path.getsize(args.img_path) / 1024
    
    # 下采样模拟
    low_res = cv2.resize(raw_img, (raw_img.shape[1]//args.scale, raw_img.shape[0]//args.scale))
    # JPEG 强力压缩打包
    _, compressed_payload = cv2.imencode('.jpg', low_res, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    
    payload_size = len(compressed_payload) / 1024
    bw_saving = (1 - payload_size / org_size) * 100
    print(f"📡 [传输链路] 原始负载 {org_size:.2f}KB -> 压缩报文 {payload_size:.2f}KB")
    print(f"💰 [传输链路] 成功节省 {bw_saving:.2f}% 网络带宽！\n")

    # 2. 【模拟边缘端】AirBox 接收并点火重建
    receiver = EdgeCloudReceiver(args.bmodel_path, device_id=args.device_id)
    low_res_img, final_4k_img = receiver.receive_and_reconstruct(compressed_payload, args)

    # 3. 保存实机结果
    os.makedirs(args.save_dir, exist_ok=True)
    low_res_save = os.path.join(args.save_dir, "node10_received_low_res.jpg")
    high_res_save = os.path.join(args.save_dir, "node10_tpu_restored_4k.jpg")
    
    cv2.imwrite(low_res_save, low_res_img)
    cv2.imwrite(high_res_save, final_4k_img)
    print("\n" + "="*55)
    print(f"🎉 物理部署验证成功！")
    print(f"📉 接收侧(低分模糊图)已保存: {low_res_save}")
    print(f"📈 算力侧(TPU高清图)已保存: {high_res_save}")
    print("="*55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Node 10: Edge-Cloud Real Deployment")
    parser.add_argument('--img_path', type=str, default='../1_data/cali_data/detection_test.jpg', help='源图像路径')
    parser.add_argument('--bmodel_path', type=str, default='../../3_models/bmodel/SwinIR_1684x_bf16.bmodel', help='SwinIR 模型路径')
    parser.add_argument('--save_dir', type=str, default='../results/node10', help='结果保存目录')
    parser.add_argument('--device_id', type=int, default=0)
    # SwinIR 核心参数
    parser.add_argument('--tile', type=int, default=192)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--scale', type=int, default=4, help='端云协作下采样倍率')
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--input_range', type=str, default='0_1')
    parser.add_argument('--tta', action='store_true')
    # 云端压缩参数
    parser.add_argument('--quality', type=int, default=20, help='传输质量 0-100 (越低越省带宽)')

    args = parser.parse_args()
    run_real_integration_test(args)