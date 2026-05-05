import numpy as np
import sophon.sail as sail
import argparse
import os
import time
import cv2

def preprocess_swinir(input_img, bmcv, model_input_w, model_input_h):
    """SwinIR预处理：缩放至模型输入尺寸并归一化"""
    # 缩放至模型输入尺寸
    resized_img = bmcv.resize(input_img, model_input_w, model_input_h)
    # 转换为浮点张量并归一化（假设输入范围为[0,255] -> 归一化到[0,1]）
    alpha_beta = (1.0/255.0, 0)  # 缩放系数和偏移量
    normalized_img = sail.BMImage(bmcv.get_handle(), model_input_h, model_input_w, 
                                sail.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_FLOAT32)
    bmcv.convert_to(resized_img, normalized_img, alpha_beta)
    return normalized_img

class SwinIR_Inference:
    def __init__(self, bmodel_path, tpu_id=0):
        # 初始化引擎
        self.engine = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSO)
        self.handle = self.engine.get_handle()
        self.graph_name = self.engine.get_graph_names()[0]
        # 输入输出配置
        self.input_name = self.engine.get_input_names(self.graph_name)[0]
        self.output_name = self.engine.get_output_names(self.graph_name)[0]
        self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
        self.output_shape = self.engine.get_output_shape(self.graph_name, self.output_name)
        # 输入尺寸
        _, self.model_h, self.model_w, _ = self.input_shape
        # 预分配输出张量
        self.output_tensor = sail.Tensor(self.handle, self.output_shape, 
                                       self.engine.get_output_dtype(self.graph_name, self.output_name), 
                                       True, True)

    def infer(self, input_tensor):
        """执行推理"""
        self.engine.process(self.graph_name, 
                           {self.input_name: input_tensor}, 
                           {self.output_name: self.output_tensor})
        return self.output_tensor.asnumpy()

def save_output_image(output_np, save_path):
    """后处理：将输出张量保存为图像"""
    # 反归一化（假设模型输出范围为[0,1] -> 转换为[0,255]
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    # NHWC -> HWC（假设batch_size=1）
    output_img = output_np.squeeze(0)
    # OpenCV保存（BGR格式）
    cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinIR超分辨率推理")
    parser.add_argument('--input_dir', type=str, required=True, help="输入图像目录")
    parser.add_argument('--bmodel_path', type=str, default="./SwinIR_1684x_f16.bmodel" required=True, help="SwinIR BModel路径")
    parser.add_argument('--device_id', type=int, default=0, help="TPU设备ID")
    parser.add_argument('--save_dir', type=str, default="./results", help="结果保存目录")
    args = parser.parse_args()

    # 初始化模型
    swinir = SwinIR_Inference(args.bmodel_path, args.device_id)
    bmcv = sail.Bmcv(swinir.handle)
    os.makedirs(args.save_dir, exist_ok=True)

    # 遍历输入图像
    valid_exts = ['.jpg', '.png', '.bmp', '.jpeg']
    image_list = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                 if os.path.splitext(f)[-1].lower() in valid_exts]
    
    if not image_list:
        print("未找到有效图像！")
        exit(1)

    for img_path in image_list:
        # 1. 读取图像
        decoder = sail.Decoder(img_path, True, args.device_id)
        bmimg = decoder.read(swinir.handle)
        # 2. 转换为RGB格式
        rgb_img = sail.BMImage(swinir.handle, bmimg.height(), bmimg.width(), 
                             sail.FORMAT_RGB_PLANAR, bmimg.dtype())
        bmcv.convert_format(bmimg, rgb_img)
        # 3. 预处理
        processed_img = preprocess_swinir(rgb_img, bmcv, swinir.model_w, swinir.model_h)
        # 4. 转换为输入张量
        input_tensor = bmcv.bm_image_to_tensor(processed_img)
        # 5. 推理
        start = time.time()
        output_np = swinir.infer(input_tensor)
        print(f"推理耗时: {(time.time()-start)*1000:.2f}ms")
        # 6. 保存结果
        save_name = os.path.join(args.save_dir, os.path.basename(img_path))
        save_output_image(output_np, save_name)
        print(f"结果已保存至: {save_name}")