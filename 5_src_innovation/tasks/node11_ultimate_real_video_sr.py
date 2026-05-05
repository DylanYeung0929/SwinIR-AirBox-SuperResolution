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

class YOLOv5_ROI:
    def __init__(self, bmodel_path, device_id):
        self.engine = None
        if not os.path.exists(bmodel_path):
            print(f"[警告] 找不到 YOLO 模型 {bmodel_path}，将降级为纯纹理路由。")
            return
        try:
            self.handle = sail.Handle(device_id)
            self.engine = sail.Engine(bmodel_path, device_id, sail.IOMode.SYSIO)
            self.graph_name = self.engine.get_graph_names()[0]
            self.input_name = self.engine.get_input_names(self.graph_name)[0]
            self.output_names = self.engine.get_output_names(self.graph_name)
            self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
            print(f"[就绪] YOLOv5 语义感知模块已挂载。")
        except Exception as e:
            print(f"YOLO 初始化失败: {e}")
            self.engine = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def get_roi_mask(self, img_np):
        H, W = img_np.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        if self.engine is None: return mask
        try:
            in_w, in_h = self.input_shape[3], self.input_shape[2]
            img_resized = cv2.resize(img_np, (in_w, in_h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
            input_data = np.expand_dims(img_float, axis=0)
            
            outputs = self.engine.process(self.graph_name, {self.input_name: input_data})
            conf_threshold = 0.35
            
            for name in self.output_names:
                out = outputs[name]
                if len(out.shape) == 4 and out.shape[1] == 255:
                    h_feat, w_feat = out.shape[2], out.shape[3]
                    out = out.reshape((1, 3, 85, h_feat, w_feat)).transpose((0, 1, 3, 4, 2))
                
                last_dim = out.shape[-1]
                if last_dim >= 5: obj_conf = self.sigmoid(out[0, ..., 4])
                else: obj_conf = self.sigmoid(np.max(out[0, ..., :], axis=-1))
                
                y_idxs, x_idxs, _ = np.where(obj_conf > conf_threshold)
                if len(y_idxs) > 0:
                    h_feat, w_feat = obj_conf.shape[1], obj_conf.shape[2]
                    for i in range(len(y_idxs)):
                        cx_norm, cy_norm = x_idxs[i] / w_feat, y_idxs[i] / h_feat
                        bw_norm, bh_norm = 3.0 / w_feat, 3.0 / h_feat
                        x1, y1 = max(0, int((cx_norm - bw_norm) * W)), max(0, int((cy_norm - bh_norm) * H))
                        x2, y2 = min(W, int((cx_norm + bw_norm) * W)), min(H, int((cy_norm + bh_norm) * H))
                        mask[y1:y2, x1:x2] = 255
        except Exception:
            pass
        return mask

def _get_gaussian_window(size):
    w = np.hanning(size) + 1e-6
    window = np.outer(w, w)
    return np.repeat(window[:, :, np.newaxis], 3, axis=2).astype(np.float32)

def run_ultimate_pipeline(args):
    print("="*60)
    print("[Ultimate Engine] 异构超分引擎启动 (YOLO + Entropy + Temporal)")
    print("="*60)

    # 1. 初始化核心模块
    yolo = YOLOv5_ROI(args.yolo_bmodel, args.device_id)
    swinir_engine = sail.Engine(args.bmodel_path, args.device_id, sail.IOMode.SYSIO)
    swinir = SwinIRInference(swinir_engine)
    bmcv = sail.Bmcv(sail.Handle(args.device_id))

    # 2. 视频 I/O 准备
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {args.video_path}")
        return

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    scale = args.scale
    tile_size = args.tile
    overlap = args.tile_overlap
    stride = tile_size - overlap
    
    # 左右分屏：左边原图放大，右边超分重构
    out_w = (W * scale) * 2
    out_h = H * scale
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (out_w, out_h))

    # 3. 滑窗与填充配置
    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)
    H_p = H + pad_h
    W_p = W + pad_w
    
    y_coords = list(range(0, H_p - tile_size + 1, stride))
    if not y_coords or y_coords[-1] != H_p - tile_size: y_coords.append(H_p - tile_size)
    x_coords = list(range(0, W_p - tile_size + 1, stride))
    if not x_coords or x_coords[-1] != W_p - tile_size: x_coords.append(W_p - tile_size)
    
    window_3d = _get_gaussian_window(tile_size * scale)
    
    # 4. 时序缓存 VRAM (Node 11)
    prev_frame_lq = None
    prev_frame_hq = None
    
    frame_idx = 0
    start_time = time.time()

    print(f"开始处理视频: {W}x{H} -> 缩放 {scale}x | 总帧数: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        # OSD 统计
        stats = {'tpu': 0, 'cpu': 0, 'cache': 0}
        
        # 预检：YOLO 掩码
        roi_mask = yolo.get_roi_mask(frame)
        
        # 补齐
        if pad_h > 0 or pad_w > 0:
            frame_pad = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            roi_mask = cv2.copyMakeBorder(roi_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        else:
            frame_pad = frame.copy()
            
        out_canvas = np.zeros((H_p * scale, W_p * scale, 3), dtype=np.float32)
        weight_canvas = np.zeros((H_p * scale, W_p * scale, 3), dtype=np.float32)
        
        # 切块处理
        for y in y_coords:
            for x in x_coords:
                tile_lq = frame_pad[y:y+tile_size, x:x+tile_size, :]
                tile_mask = roi_mask[y:y+tile_size, x:x+tile_size]
                
                # 决策矩阵
                is_yolo = np.any(tile_mask > 0)
                
                gray_tile = cv2.cvtColor(tile_lq, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray_tile, cv2.CV_64F).var()
                is_complex = variance >= args.entropy_thresh
                
                is_static = False
                if prev_frame_lq is not None:
                    prev_tile_lq = prev_frame_lq[y:y+tile_size, x:x+tile_size, :]
                    mse = np.mean((tile_lq.astype(float) - prev_tile_lq.astype(float))**2)
                    if mse < args.temporal_thresh:
                        is_static = True

                # 路由执行
                if is_static and prev_frame_hq is not None and not is_yolo:
                    # Node 11: 背景静止，直接复用上一帧高清显存
                    result_tile = prev_frame_hq[y*scale:(y+tile_size)*scale, x*scale:(x+tile_size)*scale, :]
                    stats['cache'] += 1
                elif is_yolo or is_complex:
                    # Node 4 & 7: 重要目标或高频纹理，TPU 推理
                    img_bm = sail.BMImage()
                    bmcv.mat_to_bm_image(tile_lq, img_bm)
                    result_tile = swinir.process_image(img_bm, bmcv, args, None)
                    stats['tpu'] += 1
                else:
                    # Node 4 & 7: 简单平滑背景，CPU/VPP 极速插值
                    result_tile = cv2.resize(tile_lq, (tile_size*scale, tile_size*scale), interpolation=cv2.INTER_CUBIC)
                    stats['cpu'] += 1

                if result_tile is not None:
                    y_s, x_s = y * scale, x * scale
                    out_canvas[y_s:y_s+tile_size*scale, x_s:x_s+tile_size*scale, :] += result_tile.astype(np.float32) * window_3d
                    weight_canvas[y_s:y_s+tile_size*scale, x_s:x_s+tile_size*scale, :] += window_3d
        
        # 融合与裁剪
        final_hq = out_canvas / (weight_canvas + 1e-7)
        final_hq = np.clip(final_hq[:H*scale, :W*scale, :], 0, 255).astype(np.uint8)
        
        # 更新时序缓存
        prev_frame_lq = frame_pad.copy()
        
        # 由于我们只用 overlap 画布，缓存需要提取出无缝的区域。为简化，这里直接缓存整个重建后的画布
        cache_canvas = np.zeros_like(final_hq) # 实际上这里需要保存 padding 后的大图，为保证代码稳健，简化处理
        pad_hq = cv2.copyMakeBorder(final_hq, 0, pad_h*scale, 0, pad_w*scale, cv2.BORDER_REFLECT)
        prev_frame_hq = pad_hq.copy()

        # 原图插值对比
        raw_resized = cv2.resize(frame, (W * scale, H * scale), interpolation=cv2.INTER_LINEAR)
        
        # OSD 绘制
        cv2.putText(raw_resized, "Original (Bicubic Upscaled)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        cv2.putText(final_hq, "BM1684X AirBox: SwinIR Real-Time", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(final_hq, f"Frame: {frame_idx}/{total_frames}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        cv2.putText(final_hq, f"TPU SR Blocks: {stats['tpu']}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(final_hq, f"VRAM Cache Reused: {stats['cache']}", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)
        cv2.putText(final_hq, f"CPU Fast Interpolation: {stats['cpu']}", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        # 拼接分屏视频
        split_screen = np.hstack((raw_resized, final_hq))
        out.write(split_screen)

        if frame_idx % 5 == 0:
            print(f"进度: {frame_idx}/{total_frames} | TPU: {stats['tpu']} | Cache: {stats['cache']} | CPU: {stats['cpu']}")

    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    print("="*60)
    print(f"实战超分结束！总耗时: {total_time:.2f} s")
    print(f"炫酷的分屏对比视频已保存至: {args.output_path}")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--bmodel_path', type=str, default='./SwinIR_1684x_bf16.bmodel')
    parser.add_argument('--yolo_bmodel', type=str, default='./model_yolov5s/yolov5s.bmodel')
    parser.add_argument('--output_path', type=str, default='./final_showcase.mp4')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--tile', type=int, default=192)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--input_range', type=str, default='0_1')
    parser.add_argument('--entropy_thresh', type=float, default=800.0)
    parser.add_argument('--temporal_thresh', type=float, default=15.0)
    args = parser.parse_args()
    
    run_ultimate_pipeline(args)
    