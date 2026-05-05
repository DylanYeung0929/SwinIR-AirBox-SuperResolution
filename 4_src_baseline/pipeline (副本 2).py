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

#增强超分效果 20260413 

save_lock = threading.Lock()

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.threads = []
        
        # 创建引擎和推理器
        print(f"创建引擎, 使用模型: {args.bmodel_path}")
        self.engine = []
        for _ in range(args.num_streams):
            engine = sail.Engine(args.bmodel_path, args.device_id, sail.IOMode.SYSIO)
            self.engine.append(engine)

        # 每个引擎创建独立的推理器
        self.swinir_list = [SwinIRInference(engine) for engine in self.engine]

        # 创建处理线程
        for i in range(args.num_streams):
            t = threading.Thread(target=self._run_process_thread, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)
    
    def _run_process_thread(self, thread_index):
        # 每个线程创建独立的handle和Bmcv实例
        handle = sail.Handle(self.args.device_id)
        bmcv = sail.Bmcv(handle)
        swinir = self.swinir_list[thread_index]
        print(f"处理线程 {thread_index} 初始化完成")
        
        while not self.stop_event.is_set():
            try:
                img_path = self.input_queue.get(timeout=1)
                print(f"处理线程 {thread_index} 开始处理: {img_path}")
                result = self.process_image(swinir, bmcv, img_path)
                if result:
                    self.output_queue.put(result)
                    print(f"处理线程 {thread_index} 成功处理: {img_path}")
                else:
                    print(f"处理线程 {thread_index} 处理失败: {img_path}")
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理线程 {thread_index} 异常: {e}")
                self.input_queue.task_done()

    def _get_gaussian_window(self, size):
        """核心创新点：生成二维汉明(Hann)平滑加权窗，消除分块拼接伪影"""
        w = np.hanning(size) + 1e-6 # 加上极小值防止除以0
        window = np.outer(w, w)
        return np.repeat(window[:, :, np.newaxis], 3, axis=2).astype(np.float32)

    def process_image(self, swinir, bmcv, img_path):
        try:
            # 1. 图像读取
            img_np = cv2.imread(img_path)
            if img_np is None:
                print(f"无法读取图像: {img_path}")
                return None
            
            H, W, _ = img_np.shape
            
            # --- 核心创新点：动态自适应分块机制 (Dynamic Tiling) ---
            # 直接使用模型要求的固定输入大小 192x192
            tile_size = getattr(self.args, 'tile', 192)
            if tile_size <= 0: tile_size = 192
            
            overlap = getattr(self.args, 'tile_overlap', 32)
            stride = tile_size - overlap
            scale = int(getattr(self.args, 'scale', 4))
            
            # 边缘反射 Padding，防止遇到极小图像（小于 192x192）
            pad_h = max(0, tile_size - H)
            pad_w = max(0, tile_size - W)
            if pad_h > 0 or pad_w > 0:
                img_np = cv2.copyMakeBorder(img_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                
            H_p, W_p, _ = img_np.shape
            
            # 生成分块坐标
            y_coords = list(range(0, H_p - tile_size + 1, stride))
            if y_coords[-1] != H_p - tile_size:
                y_coords.append(H_p - tile_size)
                
            x_coords = list(range(0, W_p - tile_size + 1, stride))
            if x_coords[-1] != W_p - tile_size:
                x_coords.append(W_p - tile_size)
                
            num_tiles = len(y_coords) * len(x_coords)
            if num_tiles > 1:
                print(f"触发动态分块: 原图 {H}x{W} -> 切分为 {num_tiles} 个 {tile_size}x{tile_size} 重叠区块")
            
            # --- 核心创新点：高斯融合 3D 画布 ---
            out_canvas = np.zeros((H_p * scale, W_p * scale, 3), dtype=np.float32)
            weight_canvas = np.zeros((H_p * scale, W_p * scale, 3), dtype=np.float32)
            window_3d = self._get_gaussian_window(tile_size * scale)
            
            start_time = time.time()
            
            # 逐区块进行推理
            for y in y_coords:
                for x in x_coords:
                    # 裁剪出区块
                    tile = img_np[y:y+tile_size, x:x+tile_size, :]
                    
                    # 转为 BMImage 送入底层
                    img_bm = sail.BMImage()
                    bmcv.mat_to_bm_image(tile, img_bm)
                    
                    # 硬件推理（无需保存切片文件，传 None）
                    result_tile = swinir.process_image(img_bm, bmcv, self.args, None)
                    
                    if result_tile is None:
                        print(f"区块推理失败跳过: x={x}, y={y}")
                        continue
                        
                    # 像素级高斯权重融合
                    y_s, x_s = y * scale, x * scale
                    out_canvas[y_s : y_s + tile_size*scale, x_s : x_s + tile_size*scale, :] += result_tile.astype(np.float32) * window_3d
                    weight_canvas[y_s : y_s + tile_size*scale, x_s : x_s + tile_size*scale, :] += window_3d
            
            # 归一化融合结果，并裁切回原图的目标尺度
            final_img = out_canvas / weight_canvas
            final_img = final_img[:H*scale, :W*scale, :]
            result_img = np.clip(final_img, 0, 255).astype(np.uint8)
            
            # 保存结果
            base_name = os.path.basename(img_path)
            file_name, file_ext = os.path.splitext(base_name)
            save_name = f"{file_name}_swinir{file_ext}"
            save_path = os.path.join(self.args.save_dir, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with save_lock:
                success = cv2.imwrite(save_path, result_img)
                if not success:
                    print(f"保存失败: {save_path}")
                    return None

            # 计算 PSNR 竞赛得分
            psnr = 0.0
            if getattr(self.args, 'test_y', False):
                gt_img = cv2.imread(img_path)
                if gt_img is not None and result_img is not None:
                    psnr = calculate_psnr(result_img, gt_img)
                else:
                    psnr = -1.0

            process_time = time.time() - start_time
            return {
                'path': img_path,
                'save_path': save_path,
                'psnr': psnr,
                'time': process_time
            }
            
        except Exception as e:
            print(f"处理异常: {e}")
            traceback.print_exc()
            return None

    def start(self, paths):
        for path in paths:
            self.input_queue.put(path)
        print(f"已将 {len(paths)} 张图像添加到队列")
            
        self.input_queue.join()
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=0.5)
            
        results = {}
        while not self.output_queue.empty():
            result = self.output_queue.get()
            if result:
                results[result['path']] = result
            
        return results