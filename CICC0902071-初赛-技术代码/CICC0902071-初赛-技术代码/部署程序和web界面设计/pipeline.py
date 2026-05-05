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

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.threads = []
        
        # 创建引擎和推理器
        print(f"创建引擎, 使用模型: {args.bmodel_path}")
        # self.engine = [sail.Engine(args.bmodel_path, args.device_id, sail.IOMode.SYSIO) for _ in range(args.num_streams)]
        # self.swinir_list = [SwinIRInference(engine) for engine in self.engine]
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

    def process_image(self, swinir, bmcv, img_path):
        try:
            # 使用OpenCV读取图像
            print(f"读取图像: {img_path}")
            img_np = cv2.imread(img_path)
            if img_np is None:
                print(f"无法读取图像: {img_path}")
                return None
            print(f"图像读取成功, 形状: {img_np.shape}")
                
            # 将OpenCV图像转换为BMImage - 修复参数问题
            print(f"转换OpenCV图像到BMImage...")
            # 创建临时BMImage对象
            img_bm = sail.BMImage()
            # 使用正确的参数调用 mat_to_bm_image
            bmcv.mat_to_bm_image(img_np, img_bm)  # 传递两个参数
            
            if img_bm is None:
                print(f"转换图像失败: {img_path}")
                return None
            print(f"图像转换成功")
                
            start_time = time.time()
            # 执行推理
            result_img = swinir.process_image(img_bm, bmcv, self.args, self.args.save_dir)
            
            if result_img is None:
                print(f"推理失败: {img_path}")
                return None
            print(f"推理成功")
                
            # 保存结果
            base_name = os.path.basename(img_path)
            file_name, file_ext = os.path.splitext(base_name)
            save_name = f"{file_name}_swinir{file_ext}"
            save_path = os.path.join(self.args.save_dir, save_name)

            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with save_lock:
                print(f"使用OpenCV保存结果: {save_path}")
                # 确保图像数据是整数类型
                if result_img.dtype != np.uint8:
                    result_img = result_img.astype(np.uint8)
                
                # 尝试保存图像
                try:
                    success = cv2.imwrite(save_path, result_img)
                    if success:
                        print(f"保存成功: {save_path}")
                    else:
                        print(f"保存失败: {save_path}")
                        return None
                except Exception as save_error:
                    print(f"保存图像时发生错误: {save_error}")
                    # 尝试替代保存方法
                    try:
                        # 转换为PIL格式保存
                        from PIL import Image
                        pil_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                        pil_img.save(save_path)
                        print(f"使用PIL保存成功: {save_path}")
                    except Exception as pil_error:
                        print(f"PIL保存也失败: {pil_error}")
                        return None
                            
                    
                    


            # 计算PSNR
            psnr = 0.0
            if self.args.test_y:
                gt_img = cv2.imread(img_path)
                sr_img = cv2.imread(save_path)
                if gt_img is not None and sr_img is not None:
                    psnr = calculate_psnr(sr_img, gt_img)
                    print(f"PSNR: {psnr:.2f} dB")
                else:
                    print(f"无法读取图像用于PSNR计算: {img_path} or {save_path}")
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
        # 添加所有图像到队列
        for path in paths:
            self.input_queue.put(path)
        print(f"已将 {len(paths)} 张图像添加到队列")
            
        # 等待任务完成
        self.input_queue.join()
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=0.5)
            
        # 收集结果
        results = {}
        while not self.output_queue.empty():
            result = self.output_queue.get()
            if result:
                results[result['path']] = result
            
        return results