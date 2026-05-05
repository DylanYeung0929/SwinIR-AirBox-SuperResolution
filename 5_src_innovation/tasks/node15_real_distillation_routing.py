import sophon.sail as sail
import numpy as np
import time
import os
import argparse
import cv2

# Node 15: 知识蒸馏与柔性算力调度
class RealDistillationRouter:
    def __init__(self, teacher_path, student_path, device_id=0):
        print("="*60)
        print("🚀 [Node 15] 启动 AirBox 物理级蒸馏模型路由引擎")
        self.handle = sail.Handle(device_id)
        self.bmcv = sail.Bmcv(self.handle)
        
        # 1. 物理加载教师模型 (Heavy SwinIR)
        if os.path.exists(teacher_path):
            print(f"💎 正在加载教师模型 (Heavy): {os.path.basename(teacher_path)}")
            self.teacher_engine = sail.Engine(teacher_path, device_id, sail.IOMode.SYSIO)
        else:
            self.teacher_engine = None
            print("⚠️ 警告: 找不到教师模型文件，请检查路径")

        # 2. 物理加载学生模型 (Light SwinIR - 混合精度或瘦身版)
        if os.path.exists(student_path):
            print(f"⚡ 正在加载学生模型 (Light): {os.path.basename(student_path)}")
            self.student_engine = sail.Engine(student_path, device_id, sail.IOMode.SYSIO)
        else:
            self.student_engine = None
            print("⚠️ 警告: 找不到学生模型文件")
            
        print("="*60)

    def run_inference(self, engine, img_np):
        """执行真实的 TPU 硬件推理"""
        graph_name = engine.get_graph_names()[0]
        input_name = engine.get_input_names(graph_name)[0]
        input_shape = engine.get_input_shape(graph_name, input_name)
        
        # 预处理：缩放并转为 NCHW
        img_resized = cv2.resize(img_np, (input_shape[3], input_shape[2]))
        img_f32 = img_resized.transpose((2, 0, 1)).astype(np.float32) / 255.0
        input_data = {input_name: np.expand_dims(img_f32, axis=0)}
        
        # 真实点火推理并计时
        start = time.time()
        _ = engine.process(graph_name, input_data)
        return (time.time() - start) * 1000 # 返回毫秒

    def dynamic_dispatch(self, img_path, load_threshold=200):
        """
        核心调度逻辑：
        根据当前系统的推理压力，动态决定使用哪台“发动机”
        """
        img = cv2.imread(img_path)
        if img is None: return

        print(f"\n[处理帧]: {os.path.basename(img_path)}")
        
        # 模拟上一帧的系统延迟（实际项目中可读取系统负载或上帧计时）
        last_latency = np.random.uniform(150, 350) 
        
        if last_latency > load_threshold and self.student_engine:
            print(f"🚨 系统负载过高 ({last_latency:.1f}ms > {load_threshold}ms)，触发【蒸馏降级策略】!")
            active_engine = self.student_engine
            mode = "STUDENT (Lightweight)"
        else:
            print(f"💎 系统运行平稳，维持【教师模型】高质量超分")
            active_engine = self.teacher_engine
            mode = "TEACHER (Standard)"
        
        if active_engine:
            real_time = self.run_inference(active_engine, img)
            print(f"✅ 执行结果: {mode} | 物理推理耗时: {real_time:.2f} ms")
        else:
            print("❌ 错误: 无可用推理引擎")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=str, default='./SwinIR_1684x_bf16.bmodel')
    parser.add_argument('--student', type=str, default='./SwinIR_1684x_mixed.bmodel')
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    router = RealDistillationRouter(args.teacher, args.student)
    
    # 模拟连续处理 5 帧，观察动态路由效果
    for i in range(5):
        router.dynamic_dispatch(args.image)