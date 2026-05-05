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

#动态路由的信息熵阈值:对大面积相同颜色的区域选用更快的超分方法,加快运算速度 20260415

save_lock = threading.Lock()

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.threads = []
        
        # 创新点参数：动态路由的信息熵阈值
        # 方差小于该值被认为是平滑背景（如天空/白墙）
        self.entropy_threshold = getattr(args, 'entropy_threshold', 800.0)
        
        print(f"创建引擎, 使用模型: {args.bmodel_path}")
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
        w = np.hanning(size) + 1e-6
        window = np.outer(w, w)
        return np.repeat(window[:, :, np.newaxis], 3, axis=2).astype(np.float32)

    def process_image(self, swinir, bmcv, img_path, thread_index):
        try:
            img_np = cv2.imread(img_path)
            if img_np is None: return None
            H, W, _ = img_np.shape
            
            tile_size = getattr(self.args, 'tile', 192)
            if tile_size <= 0: tile_size = 192
            
            overlap = getattr(self.args, 'tile_overlap', 32)
            stride = tile_size - overlap
            scale = int(getattr(self.args, 'scale', 4))
            
            pad_h = max(0, tile_size - H)
            pad_w = max(0, tile_size - W)
            if pad_h > 0 or pad_w > 0:
                img_np = cv2.copyMakeBorder(img_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                
            H_p, W_p, _ = img_np.shape
            
            y_coords = list(range(0, H_p - tile_size + 1, stride))
            if y_coords[-1] != H_p - tile_size: y_coords.append(H_p - tile_size)
            x_coords = list(range(0, W_p - tile_size + 1, stride))
            if x_coords[-1] != W_p - tile_size: x_coords.append(W_p - tile_size)
                
            num_tiles = len(y_coords) * len(x_coords)
            
            out_canvas = np.zeros((H_p * scale, W_p * scale, 3), dtype=np.float32)
            weight_canvas = np.zeros((H_p * scale, W_p * scale, 3), dtype=np.float32)
            window_3d = self._get_gaussian_window(tile_size * scale)
            
            start_time = time.time()
            routed_to_tpu = 0
            routed_to_cpu = 0
            
            # --- 创新点：逐区块进行信息熵路由 ---
            for y in y_coords:
                for x in x_coords:
                    tile = img_np[y:y+tile_size, x:x+tile_size, :]
                    
                    # 1. 计算图像块复杂度 (Laplacian Variance)
                    gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                    variance = cv2.Laplacian(gray_tile, cv2.CV_64F).var()
                    
                    # 2. 动态路由决策
                    if variance < self.entropy_threshold:
                        # 平滑区域：直接走极速通道 (Bicubic 插值)，零 TPU 消耗
                        # print(f"  [路由] 块方差 {variance:.1f} < {self.entropy_threshold} -> 极速通道 (CPU插值)")
                        result_tile = cv2.resize(tile, (tile_size*scale, tile_size*scale), interpolation=cv2.INTER_CUBIC)
                        routed_to_cpu += 1
                    else:
                        # 复杂区域：走重度超分通道 (TPU SwinIR)
                        # print(f"  [路由] 块方差 {variance:.1f} >= {self.entropy_threshold} -> 重度超分 (TPU SwinIR)")
                        img_bm = sail.BMImage()
                        bmcv.mat_to_bm_image(tile, img_bm)
                        result_tile = swinir.process_image(img_bm, bmcv, self.args, None)
                        routed_to_tpu += 1
                        
                    if result_tile is None: continue
                        
                    y_s, x_s = y * scale, x * scale
                    out_canvas[y_s : y_s + tile_size*scale, x_s : x_s + tile_size*scale, :] += result_tile.astype(np.float32) * window_3d
                    weight_canvas[y_s : y_s + tile_size*scale, x_s : x_s + tile_size*scale, :] += window_3d
            
            print(f"[{os.path.basename(img_path)}] 动态路由统计: 复杂纹理块(TPU) {routed_to_tpu} 个, 平滑背景块(极速) {routed_to_cpu} 个")
            
            final_img = out_canvas / weight_canvas
            final_img = final_img[:H*scale, :W*scale, :]
            result_img = np.clip(final_img, 0, 255).astype(np.uint8)
            
            base_name = os.path.basename(img_path)
            file_name, file_ext = os.path.splitext(base_name)
            
            # --- 修改点：引入版本号隔离，防止覆盖旧版结果 ---
            # 旧版: save_name = f"{file_name}_swinir{file_ext}"
            # 新版: 添加 _v4_routing 后缀
            save_name = f"{file_name}_swinir_v4_routing{file_ext}"
            
            save_path = os.path.join(self.args.save_dir, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with save_lock:
                cv2.imwrite(save_path, result_img)

            psnr = 0.0
            if getattr(self.args, 'test_y', False):
                gt_img = cv2.imread(img_path)
                if gt_img is not None and result_img is not None:
                    psnr = calculate_psnr(result_img, gt_img)
                else: psnr = -1.0

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
        for path in paths: self.input_queue.put(path)
        self.input_queue.join()
        self.stop_event.set()
        for t in self.threads: t.join(timeout=0.5)
            
        results = {}
        while not self.output_queue.empty():
            result = self.output_queue.get()
            if result: results[result['path']] = result
        return results