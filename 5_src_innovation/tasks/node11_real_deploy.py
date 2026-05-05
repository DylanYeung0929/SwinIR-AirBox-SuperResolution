
import cv2
import numpy as np
import sophon.sail as sail
import os
import time
import argparse

try:
    from core.inference import SwinIRInference
except ImportError:
    from inference import SwinIRInference

class TemporalReuseEngine:
    def __init__(self, bmodel_path, device_id=0, threshold=10.0):
        print(f"🚀 [AirBox] 启动时序稀疏复用引擎 (Threshold: {threshold})")
        self.handle = sail.Handle(device_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.engine = sail.Engine(bmodel_path, device_id, sail.IOMode.SYSIO)
        self.swinir = SwinIRInference(self.engine)
        self.threshold = threshold
        self.prev_frame_lq = None
        self.prev_frame_hq = None

    def process_frame(self, frame_lq, args):
        h, w = frame_lq.shape[:2]
        tile_size = args.tile
        scale = args.scale
        out_h, out_w = h * scale, w * scale
        final_hq = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        stats = {'reused': 0, 'tpu': 0}
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                patch_lq = frame_lq[y:y+tile_size, x:x+tile_size]
                is_reusable = False
                if self.prev_frame_lq is not None:
                    prev_patch_lq = self.prev_frame_lq[y:y+tile_size, x:x+tile_size]
                    diff = np.mean((patch_lq.astype(float) - prev_patch_lq.astype(float))**2)
                    if diff < self.threshold:
                        is_reusable = True
                
                if is_reusable:
                    stats['reused'] += 1
                    final_hq[y*scale:(y+tile_size)*scale, x*scale:(x+tile_size)*scale] = \
                        self.prev_frame_hq[y*scale:(y+tile_size)*scale, x*scale:(x+tile_size)*scale]
                else:
                    stats['tpu'] += 1
                    img_bm = sail.BMImage()
                    self.bmcv.mat_to_bm_image(patch_lq, img_bm)
                    res_tile = self.swinir.process_image(img_bm, self.bmcv, args, None)
                    final_hq[y*scale:(y+tile_size)*scale, x*scale:(x+tile_size)*scale] = res_tile
        
        self.prev_frame_lq = frame_lq.copy()
        self.prev_frame_hq = final_hq.copy()
        return final_hq, stats

def run_node11_test(args):
    engine = TemporalReuseEngine(args.bmodel_path, args.device_id, args.threshold)
    raw_img = cv2.imread(args.img_path)
    if raw_img is None: return
    
    frame1 = cv2.resize(raw_img, (192*2, 192*2))
    frame2 = frame1.copy()
    frame2[192:, 192:] = cv2.GaussianBlur(frame2[192:, 192:], (5,5), 0)
    
    print("\n--- 处理第一帧 (全量) ---")
    engine.process_frame(frame1, args)
    
    print("\n--- 处理第二帧 (复用激活并存图) ---")
    final_hq, s2 = engine.process_frame(frame2, args)
    
    # 🌟 关键修复：保存物理图片
    save_path = "node11_tpu_restored_4k.jpg"
    cv2.imwrite(save_path, final_hq)
    print(f"✅ 结果已保存至: {save_path}")
    print(f"🚀 FPS 收益: {(s2['reused'] + s2['tpu']) / s2['tpu']:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./detection_test.jpg')
    parser.add_argument('--bmodel_path', type=str, default='./SwinIR_1684x_bf16.bmodel')
    parser.add_argument('--threshold', type=float, default=10.0)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--tile', type=int, default=192)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--input_range', type=str, default='0_1')
    args = parser.parse_args()
    run_node11_test(args)