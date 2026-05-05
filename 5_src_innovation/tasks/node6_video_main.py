import cv2
import time
import argparse
import os
import threading
import queue
import numpy as np
import sophon.sail as sail
import traceback
from inference import SwinIRInference

# ==========================================
# YOLOv5 多头自适应提取器 (内存版)
# ==========================================
class YOLOv5_ROI:
    def __init__(self, bmodel_path, device_id):
        self.engine = None
        if not os.path.exists(bmodel_path):
            print(f"⚠️ YOLO 模块未激活: 找不到模型 {bmodel_path}")
            return
            
        try:
            print(f"✅ 挂载 YOLOv5 级联大脑: {bmodel_path}")
            self.handle = sail.Handle(device_id)
            self.engine = sail.Engine(bmodel_path, device_id, sail.IOMode.SYSIO)
            self.graph_name = self.engine.get_graph_names()[0]
            self.input_name = self.engine.get_input_names(self.graph_name)[0]
            self.output_names = self.engine.get_output_names(self.graph_name)
            self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
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
            
            input_tensor = {self.input_name: input_data}
            outputs = self.engine.process(self.graph_name, input_tensor)
            conf_threshold = 0.4
            
            for name in self.output_names:
                out = outputs[name]
                if len(out.shape) == 4 and out.shape[1] == 255:
                    h_feat, w_feat = out.shape[2], out.shape[3]
                    out = out.reshape((1, 3, 85, h_feat, w_feat)).transpose((0, 1, 3, 4, 2))
                
                last_dim = out.shape[-1]
                if last_dim >= 5:
                    obj_conf = self.sigmoid(out[0, ..., 4])
                else:
                    obj_conf = self.sigmoid(np.max(out[0, ..., :], axis=-1))
                
                y_idxs, x_idxs, _ = np.where(obj_conf > conf_threshold)
                if len(y_idxs) > 0:
                    h_feat, w_feat = obj_conf.shape[1], obj_conf.shape[2]
                    for i in range(len(y_idxs)):
                        cx_norm = x_idxs[i] / w_feat
                        cy_norm = y_idxs[i] / h_feat
                        box_w_norm = 2.5 / w_feat
                        box_h_norm = 2.5 / h_feat
                        
                        x1 = int((cx_norm - box_w_norm) * W)
                        y1 = int((cy_norm - box_h_norm) * H)
                        x2 = int((cx_norm + box_w_norm) * W)
                        y2 = int((cy_norm + box_h_norm) * H)
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W, x2), min(H, y2)
                        mask[y1:y2, x1:x2] = 255
        except Exception:
            pass
        return mask

# ==========================================
# 视频帧处理工作者线程
# ==========================================
def _get_gaussian_window(size):
    w = np.hanning(size) + 1e-6
    window = np.outer(w, w)
    return np.repeat(window[:, :, np.newaxis], 3, axis=2).astype(np.float32)

def worker_thread(in_queue, out_queue, swinir, bmcv, yolo_detector, args, thread_id):
    print(f"✅ 视频超分并发槽位 [{thread_id}] 已就绪")
    
    tile_size = args.tile
    overlap = args.tile_overlap
    stride = tile_size - overlap
    scale = args.scale
    window_3d = _get_gaussian_window(tile_size * scale)
    
    while True:
        task = in_queue.get()
        if task is None: # 毒药丸机制，安全退出
            in_queue.task_done()
            break
            
        frame_id, frame = task
        H, W, _ = frame.shape
        
        # 1. YOLO预检
        roi_mask = yolo_detector.get_roi_mask(frame)
        
        # 2. Padding
        pad_h = max(0, tile_size - H)
        pad_w = max(0, tile_size - W)
        if pad_h > 0 or pad_w > 0:
            frame = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            roi_mask = cv2.copyMakeBorder(roi_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            
        H_p, W_p, _ = frame.shape
        y_coords = list(range(0, H_p - tile_size + 1, stride))
        if not y_coords or y_coords[-1] != H_p - tile_size: y_coords.append(H_p - tile_size)
        x_coords = list(range(0, W_p - tile_size + 1, stride))
        if not x_coords or x_coords[-1] != W_p - tile_size: x_coords.append(W_p - tile_size)
            
        out_canvas = np.zeros((H_p * scale, W_p * scale, 3), dtype=np.float32)
        weight_canvas = np.zeros((H_p * scale, W_p * scale, 3), dtype=np.float32)
        
        start_time = time.time()
        stats = {'yolo': 0, 'entropy': 0, 'cpu': 0}
        
        # 3. 双脑分流执行
        for y in y_coords:
            for x in x_coords:
                tile = frame[y:y+tile_size, x:x+tile_size, :]
                tile_mask = roi_mask[y:y+tile_size, x:x+tile_size]
                
                if np.any(tile_mask > 0):
                    stats['yolo'] += 1
                    img_bm = sail.BMImage()
                    bmcv.mat_to_bm_image(tile, img_bm)
                    result_tile = swinir.process_image(img_bm, bmcv, args, None)
                else:
                    gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                    variance = cv2.Laplacian(gray_tile, cv2.CV_64F).var()
                    if variance >= getattr(args, 'entropy_threshold', 800.0):
                        stats['entropy'] += 1
                        img_bm = sail.BMImage()
                        bmcv.mat_to_bm_image(tile, img_bm)
                        result_tile = swinir.process_image(img_bm, bmcv, args, None)
                    else:
                        stats['cpu'] += 1
                        result_tile = cv2.resize(tile, (tile_size*scale, tile_size*scale), interpolation=cv2.INTER_CUBIC)
                    
                if result_tile is not None:
                    y_s, x_s = y * scale, x * scale
                    out_canvas[y_s:y_s+tile_size*scale, x_s:x_s+tile_size*scale, :] += result_tile.astype(np.float32) * window_3d
                    weight_canvas[y_s:y_s+tile_size*scale, x_s:x_s+tile_size*scale, :] += window_3d
        
        final_img = out_canvas / (weight_canvas + 1e-7)
        final_img = final_img[:H*scale, :W*scale, :]
        result_img = np.clip(final_img, 0, 255).astype(np.uint8)
        
        cost_time = time.time() - start_time
        
        # 将结果送回有序写入队列
        out_queue.put({
            'frame_id': frame_id,
            'image': result_img,
            'stats': stats,
            'time': cost_time
        })
        in_queue.task_done()

# ==========================================
# 视频流控制主中枢
# ==========================================
def run_video_sr(args):
    print("🚀 [初始化] 异构多线程视频超分引擎启动...")
    
    # 初始化组件
    yolo_detector = YOLOv5_ROI(args.yolo_bmodel, args.device_id)
    engines = [sail.Engine(args.bmodel_path, args.device_id, sail.IOMode.SYSIO) for _ in range(args.num_streams)]
    swinir_list = [SwinIRInference(engine) for engine in engines]
    bmcv = sail.Bmcv(sail.Handle(args.device_id))
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"❌ 错误: 无法打开视频 {args.video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 视频参数: 分辨率 {width}x{height} | 帧率 {fps} | 总帧数 {total_frames}")

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "output_4k_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width*args.scale, height*args.scale))

    # 线程与队列
    in_queue = queue.Queue(maxsize=args.num_streams * 2)
    out_queue = queue.Queue()
    threads = []
    
    for i in range(args.num_streams):
        t = threading.Thread(target=worker_thread, args=(in_queue, out_queue, swinir_list[i], bmcv, yolo_detector, args, i))
        t.daemon = True
        t.start()
        threads.append(t)

    print("🎬 引擎全功率运转，开始实时超分推流...")
    
    # 为了保证多线程下视频帧不乱序，引入缓冲机制
    frame_buffer = {}
    next_write_id = 0
    read_frame_id = 0
    
    start_time = time.time()
    
    # 读帧子线程，防止网络/硬盘 I/O 阻塞 TPU 计算
    def frame_reader():
        nonlocal read_frame_id
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            in_queue.put((read_frame_id, frame))
            read_frame_id += 1
        # 发送毒药丸结束计算线程
        for _ in range(args.num_streams):
            in_queue.put(None)
            
    reader_thread = threading.Thread(target=frame_reader)
    reader_thread.start()

    # 主线程负责按顺序写入视频
    while next_write_id < total_frames:
        try:
            # 获取处理完的帧数据
            res = out_queue.get(timeout=5)
            frame_buffer[res['frame_id']] = res
            
            # 按顺序出库写入视频
            while next_write_id in frame_buffer:
                data = frame_buffer.pop(next_write_id)
                out.write(data['image'])
                s = data['stats']
                print(f"✨ 写入第 [{next_write_id+1}/{total_frames}] 帧 | YOLO:{s['yolo']} 纹理:{s['entropy']} CPU:{s['cpu']} | 耗时:{data['time']:.2f}s")
                next_write_id += 1
                
        except queue.Empty:
            if not reader_thread.is_alive() and in_queue.empty() and out_queue.empty():
                break # 全部处理完毕
                
    cap.release()
    out.release()
    total_time = time.time() - start_time
    print(f"✅ 任务大捷！总耗时: {total_time:.2f}s | 生成的高清视频已保存至: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help='输入老旧/低清视频路径')
    parser.add_argument('--bmodel_path', type=str, default='../3_models/bmodel/SwinIR_1684x_bf16.bmodel')
    parser.add_argument('--yolo_bmodel', type=str, default='../3_models/bmodel/yolov5s/yolov5s.bmodel')
    parser.add_argument('--save_dir', type=str, default='./results_video')
    parser.add_argument('--num_streams', type=int, default=4, help='并发路数')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--tile', type=int, default=192)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--scale', type=int, default=4)
    
    # [新增] 补齐 SwinIR 推理底层需要的参数，防止报错崩溃
    parser.add_argument('--input_range', type=str, default='0_1', help='输入像素范围 0_1 或 -1_1')
    parser.add_argument('--entropy_threshold', type=float, default=800.0, help='信息熵阈值')
    parser.add_argument('--tta', action='store_true', help='开启 TTA (测试时增强)')
    parser.add_argument('--test_y', action='store_true', help='仅评估 Y 通道 PSNR')
    parser.add_argument('--window_size', type=int, default=8, help='SwinIR 窗口大小')

    args = parser.parse_args()
    
    run_video_sr(args)