import numpy as np
import time
import threading
from utils import preprocess, postprocess
engine_lock = threading.Lock()

class SwinIRInference:
    def __init__(self, engine):
        self.engine = engine
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]
        self.output_name = self.engine.get_output_names(self.graph_name)[0]
        self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
        self.output_shape = self.engine.get_output_shape(self.graph_name, self.output_name)
        print(f"模型输入形状: {self.input_shape}, 输出形状: {self.output_shape}")

    def process_image(self, img_bm, bmcv, args, save_dir=None):
        try:
            # 预处理
            print("开始预处理...")
            input_tensor = preprocess(img_bm, bmcv, args.input_range, self.input_shape)
            if input_tensor is None:
                print("预处理失败: 返回空张量")
                return None
            print(f"预处理成功, 输入张量形状: {input_tensor.shape}")
                
            # 执行推理
            start_time = time.time()
            with engine_lock:
                print("开始推理...")
                outputs = self.engine.process(self.graph_name, {self.input_name: input_tensor})
            infer_time = time.time() - start_time
            print(f"推理完成, 耗时: {infer_time:.4f}秒")
            
            # 获取输出
            output_tensor = outputs[self.output_name]
            output_np = np.copy(output_tensor)
            print(f"推理输出形状: {output_np.shape}")
            
            # 后处理
            print("开始后处理...")
            result_img = postprocess(output_np, bmcv, args.input_range, self.output_shape)
            if result_img is None:
                print("后处理失败: 返回空图像")
            else:
                print("后处理成功")
            return result_img
            
        except Exception as e:
            print(f"推理异常: {e}")
            return None
        # if not img_paths:
        #     return []
            
        # # 准备输入张量列表
        # input_tensors = []
        # for path in img_paths:
        #     tensor = preprocess(path, bmcv, args.input_range)
        #     if tensor is None:
        #         print(f"预处理失败: {path}")
        #         continue
        #     input_tensors.append(tensor)
        
        # if not input_tensors:
        #     return []
            
        # # 合并为批次
        # batch_input = np.concatenate(input_tensors, axis=0)
        # batch_size = batch_input.shape[0]
        # print(f"批处理大小: {batch_size}")
        
        # # 创建输入张量
        # input_shape = list(self.input_shape)
        # input_shape[0] = batch_size  # 更新批次大小
        
        # # 确保输入尺寸匹配
        # if batch_input.shape != tuple(input_shape):
        #     # 尝试重塑
        #     if batch_input.size == np.prod(input_shape):
        #         batch_input = batch_input.reshape(input_shape)
        #         print(f"重塑输入形状为: {batch_input.shape}")
        #     else:
        #         print("无法重塑输入数据！跳过此批次。")
        #         return []
        
        # # 执行推理
        # start_time = time.time()
        # outputs = self.engine.process(self.graph_name, {self.input_name: batch_input})
        # infer_time = time.time() - start_time
        # print(f"批量推理时间: {infer_time:.4f}秒 (平均每张图像 {infer_time/batch_size:.4f}秒)")
        
        # # 获取输出
        # output_tensor = outputs[self.output_name]
        # output_np = np.array(output_tensor)
        
        # # 处理每张图像的结果
        # results = []
        # for i in range(batch_size):
        #     # 取出当前图像对应的输出
        #     img_out = output_np[i]
        #     # 后处理
        #     result_img = postprocess(img_out, bmcv, args.input_range)
        #     if result_img is None:
        #         print(f"后处理失败: {img_paths[i]}")
        #         continue
        #     results.append((img_paths[i], result_img))
            
        # return results