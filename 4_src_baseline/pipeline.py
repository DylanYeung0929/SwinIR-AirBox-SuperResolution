import os
import time
import threading
import queue
import numpy as np
import sophon.sail as sail
import cv2
import traceback
from utils import calculate_psnr
from inference import SwinIRInference

save_lock = threading.Lock()

class YOLOv5_ROI:
    def __init__(self, bmodel_path, device_id):
        self.engine = None
        if not os.path.exists(bmodel_path):
            print(f"YOLO 模块未激活: 找不到模型 {bmodel_path}")
            return
            
        try:
            print(f"挂载 YOLOv5 级联引擎: {bmodel_path}")
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
        """数值稳定的 Sigmoid 函数"""
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def get_roi_mask(self, img_np):
        """
        运行 YOLO 并返回一个二维的二值掩码 (Mask)
        ROI 区域值为 255，背景为 0
        """
        H, W = img_np.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        
        if self.engine is None:
            return mask
            
        try:
            # 1. 预处理 (适配模型输入尺寸)
            in_w, in_h = self.input_shape[3], self.input_shape[2]
            img_resized = cv2.resize(img_np, (in_w, in_h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
            input_data = np.expand_dims(img_float, axis=0)
            
            # 2. 推理
            input_tensor = {self.input_name: input_data}
            outputs = self.engine.process(self.graph_name, input_tensor)
            
            # 3. 多头输出解析 (针对 350, 498, 646 这种未合并的节点)
            # 设定置信度阈值，我们不需要极高的准确度，只要能捕捉到大概位置即可
            conf_threshold = 0.4
            
            for name in self.output_names:
                out = outputs[name] # 典型 Shape: [1, 3, H_feat, W_feat, 85] 或 [1, 255, H_feat, W_feat]
                
                # 兼容性处理：如果是三输出未合并的 [1, 255, h, w]
                if len(out.shape) == 4 and out.shape[1] == 255:
                    h_feat, w_feat = out.shape[2], out.shape[3]
                    # 重塑为 [1, 3, 85, h, w] 并置换为 [1, 3, h, w, 85]
                    out = out.reshape((1, 3, 85, h_feat, w_feat)).transpose((0, 1, 3, 4, 2))
                
                # 获取 Objectness Score (通常在最后一维的索引 4)
                last_dim = out.shape[-1]
                if last_dim >= 5:
                    obj_conf = self.sigmoid(out[0, ..., 4])
                else:
                    obj_conf = self.sigmoid(np.max(out[0, ..., :], axis=-1))
                
                # 找到所有高于阈值的特征格点
                y_idxs, x_idxs, anchor_idxs = np.where(obj_conf > conf_threshold)
                
                if len(y_idxs) > 0:
                    h_feat, w_feat = obj_conf.shape[1], obj_conf.shape[2]
                    for i in range(len(y_idxs)):
                        # 将特征图坐标映射回原图归一化坐标
                        cx_norm = x_idxs[i] / w_feat
                        cy_norm = y_idxs[i] / h_feat
                        
                        # 设定一个大概的感知区域（对应特征图上的感受野）
                        # 步长越小（特征图越大），感知的框越小
                        box_w_norm = 2.5 / w_feat
                        box_h_norm = 2.5 / h_feat
                        
                        x1 = int((cx_norm - box_w_norm) * W)
                        y1 = int((cy_norm - box_h_norm) * H)
                        x2 = int((cx_norm + box_w_norm) * W)
                        y2 = int((cy_norm + box_h_norm) * H)
                        
                        # 边界裁剪并涂抹掩码
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W, x2), min(H, y2)
                        mask[y1:y2, x1:x2] = 255
                        
        except Exception as e:
            # 记录错误但不中断主流程
            # print(f"YOLO 解析细节警告: {e}")
            pass
            
        return mask

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.threads = []
        
        # 创新点参数：动态路由的信息熵阈值
        self.entropy_threshold = getattr(args, 'entropy_threshold', 800.0)
        
        # 挂载 YOLO 级联检测器
        yolo_path = getattr(args, 'yolo_bmodel', './model_yolov5s/yolov5s.bmodel')
        self.yolo_detector = YOLOv5_ROI(yolo_path, args.device_id)
        
        print(f"创建引擎, 使用模型: {args.bmodel_path}")
        # 初始化 SwinIR 异构引擎
        self.engine = []
        for _ in range(args.num_streams):
            engine = sail.Engine(args.bmodel_path, args.device_id, sail.IOMode.SYSIO)
            self.engine.append(engine)

        self.swinir_list = [SwinIRInference(engine) for engine in self.engine]

        for i in range(args.num_streams):
            t = threading.Thread(target=self._run_process_thread, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)
    
    def _run_process_thread(self, thread_index):
        handle = sail.Handle(self.args.device_id)
        bmcv = sail.Bmcv(handle)
        swinir = self.swinir_list[thread_index]
        print(f"处理线程 {thread_index} 初始化完成")
        
        while not self.stop_event.is_set():
            try:
                img_path = self.input_queue.get(timeout=1)
                result = self.process_image(swinir, bmcv, img_path, thread_index)
                if result:
                    self.output_queue.put(result)
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理线程 {thread_index} 异常: {e}")
                self.input_queue.task_done()

    def _get_gaussian_window(self, size):
        """生成高斯平滑窗，用于分块融合"""
        w = np.hanning(size) + 1e-6
        window = np.outer(w, w)
        return np.repeat(window[:, :, np.newaxis], 3, axis=2).astype(np.float32)

    def process_image(self, swinir, bmcv, img_path, thread_index):
        try:
            img_np = cv2.imread(img_path)
            if img_np is None: return None
            H, W, _ = img_np.shape
            
            # 1. 语义预检：获取整图的 ROI 掩码
            roi_mask = self.yolo_detector.get_roi_mask(img_np)
            
            tile_size = getattr(self.args, 'tile', 192)
            overlap = getattr(self.args, 'tile_overlap', 32)
            stride = tile_size - overlap
            scale = int(getattr(self.args, 'scale', 4))
            
            # 边界补齐 (Padding)
            pad_h = max(0, tile_size - H)
            pad_w = max(0, tile_size - W)
            if pad_h > 0 or pad_w > 0:
                img_np = cv2.copyMakeBorder(img_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                roi_mask = cv2.copyMakeBorder(roi_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                
            H_p, W_p, _ = img_np.shape
            
            # 生成分块坐标
            y_coords = list(range(0, H_p - tile_size + 1, stride))
            if not y_coords or y_coords[-1] != H_p - tile_size: y_coords.append(H_p - tile_size)
            x_coords = list(range(0, W_p - tile_size + 1, stride))
            if not x_coords or x_coords[-1] != W_p - tile_size: x_coords.append(W_p - tile_size)
                
            out_canvas = np.zeros((H_p * scale, W_p * scale, 3), dtype=np.float32)
            weight_canvas = np.zeros((H_p * scale, W_p * scale, 3), dtype=np.float32)
            window_3d = self._get_gaussian_window(tile_size * scale)
            
            start_time = time.time()
            stats = {'yolo': 0, 'entropy': 0, 'cpu': 0}
            
            # --- 核心：双重感知动态路由 ---
            for y in y_coords:
                for x in x_coords:
                    tile = img_np[y:y+tile_size, x:x+tile_size, :]
                    tile_mask = roi_mask[y:y+tile_size, x:x+tile_size]
                    
                    # 路由决策 1: YOLO 是否发现重要语义目标？
                    is_semantic_target = np.any(tile_mask > 0)
                    
                    if is_semantic_target:
                        # 语义命中 -> 强制执行 TPU 深度超分
                        stats['yolo'] += 1
                        img_bm = sail.BMImage()
                        bmcv.mat_to_bm_image(tile, img_bm)
                        result_tile = swinir.process_image(img_bm, bmcv, self.args, None)
                    else:
                        # 路由决策 2: 纹理复杂度判定
                        gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                        variance = cv2.Laplacian(gray_tile, cv2.CV_64F).var()
                        
                        if variance >= self.entropy_threshold:
                            # 复杂纹理 -> 执行 TPU 推理
                            stats['entropy'] += 1
                            img_bm = sail.BMImage()
                            bmcv.mat_to_bm_image(tile, img_bm)
                            result_tile = swinir.process_image(img_bm, bmcv, self.args, None)
                        else:
                            # 纯平滑背景 -> 执行 CPU 极速插值
                            stats['cpu'] += 1
                            result_tile = cv2.resize(tile, (tile_size*scale, tile_size*scale), interpolation=cv2.INTER_CUBIC)
                        
                    if result_tile is not None:
                        y_s, x_s = y * scale, x * scale
                        # 加权融合到画布
                        out_canvas[y_s:y_s+tile_size*scale, x_s:x_s+tile_size*scale, :] += result_tile.astype(np.float32) * window_3d
                        weight_canvas[y_s:y_s+tile_size*scale, x_s:x_s+tile_size*scale, :] += window_3d
            
            print(f"[{os.path.basename(img_path)}] 路由统计: YOLO命中(TPU) {stats['yolo']} | 纹理命中(TPU) {stats['entropy']} | 平滑背景{stats['cpu']}")
            
            # 最终融合输出
            final_img = out_canvas / (weight_canvas + 1e-7)
            final_img = final_img[:H*scale, :W*scale, :]
            result_img = np.clip(final_img, 0, 255).astype(np.uint8)
            
            # 保存结果 (V5 版本)
            base_name = os.path.basename(img_path)
            file_name, file_ext = os.path.splitext(base_name)
            save_name = f"{file_name}_swinir_v5_yolo{file_ext}"
            save_path = os.path.join(self.args.save_dir, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with save_lock:
                cv2.imwrite(save_path, result_img)

            psnr = 0.0
            if getattr(self.args, 'test_y', False):
                gt_img = cv2.imread(img_path)
                if gt_img is not None:
                    psnr = calculate_psnr(result_img, gt_img)

            return {
                'path': img_path,
                'save_path': save_path,
                'psnr': psnr,
                'time': time.time() - start_time
            }
            
        except Exception:
            traceback.print_exc()
            return None

    def start(self, paths):
        for path in paths: self.input_queue.put(path)
        self.input_queue.join()
        self.stop_event.set()
        for t in self.threads: t.join(timeout=0.5)
            
        results = {}
        while not self.output_queue.empty():
            res = self.output_queue.get()
            if res: results[res['path']] = res
        return results