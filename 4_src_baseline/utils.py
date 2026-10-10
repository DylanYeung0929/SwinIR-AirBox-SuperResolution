import numpy as np
import sophon.sail as sail
import cv2
import traceback

def preprocess(img_bm, bmcv, input_range='0_1', model_input_shape=None):
    try:
        # 获取目标尺寸
        if model_input_shape is None:
            target_w, target_h = 192, 192
        else:
            # 模型输入形状: [N, C, H, W]
            target_w, target_h = model_input_shape[3], model_input_shape[2]
            
        # 1. 硬件缩放 (VPP)
        # 直接在硬件申请预留空间，避免重复分配
        resized_bm = sail.BMImage(bmcv.get_handle(), target_h, target_w,
                                  sail.Format.FORMAT_BGR_PLANAR, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
        bmcv.vpp_resize(img_bm, resized_bm, target_w, target_h)
            
        # 2. 硬件色彩转换 (VPP: BGR -> RGB)
        rgb_bm = sail.BMImage(bmcv.get_handle(), target_h, target_w,
                              sail.Format.FORMAT_RGB_PLANAR, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
        bmcv.convert_format(resized_bm, rgb_bm)
        
        # 3. 硬件量化与数据类型转换 (VPP: uint8 -> float32 且归一化)
        # 将原本由 CPU Numpy 承担的除以 255 运算，转化为硬件层的乘加运算
        rgb_f32_bm = sail.BMImage(bmcv.get_handle(), target_h, target_w,
                                  sail.Format.FORMAT_RGB_PLANAR, sail.ImgDtype.DATA_TYPE_EXT_FLOAT32)
        
        if input_range == '0_1':
            a = 1.0 / 255.0
            b = 0.0
        else:
            a = 2.0 / 255.0
            b = -1.0
        converto_attr = ((a, b), (a, b), (a, b))
        
        # 调用硬件完成归一化
        bmcv.convert_to(rgb_bm, rgb_f32_bm, converto_attr)

        # 4. 零拷贝获取 Tensor
        tensor = bmcv.bm_image_to_tensor(rgb_f32_bm)
        
        # 返回 numpy 用于兼容主程序的 pipeline 逻辑
        # 此时已经省去了最耗时的 CPU 端 float32 除法大循环
        img_data = tensor.asnumpy()
        if len(img_data.shape) == 3:
            img_data = np.expand_dims(img_data, axis=0)
            
        return img_data

    except Exception as e:
        print(f"预处理异常: {e}")
        traceback.print_exc()
        return None

def postprocess(output_np, bmcv, input_range='0_1', model_output_shape=None):
    try:
        # output_np Shape: (1, 3, H, W)
        if len(output_np.shape) == 4:
            output_np = output_np[0] 
            
        #In-place 原地计算，降低内存分配开销与垃圾回收压力
        if input_range == '0_1':
            output_np *= 255.0
        else:
            output_np += 1.0
            output_np *= 127.5
            
        # 截断并转换类型 (out=output_np 继续保持原地覆盖)
        np.clip(output_np, 0, 255, out=output_np)
        img_data = output_np.astype(np.uint8)

        # CHW -> HWC
        img_data = np.transpose(img_data, (1, 2, 0))

        # RGB -> BGR (底层使用高度优化的 C++ 代码)
        if img_data.shape[2] == 3:
            bgr_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        else:
            bgr_img = img_data
            
        return bgr_img
        
    except Exception as e:
        print(f"后处理异常: {e}")
        traceback.print_exc()
        return None
 
def calculate_psnr(sr_img, gt_img):
    if sr_img is None or gt_img is None:
        return -1.0
    
    try:
        # 确保图像尺寸相同
        if sr_img.shape != gt_img.shape:
            gt_img = cv2.resize(gt_img, (sr_img.shape[1], sr_img.shape[0]))
            
        # 转换为YCrCb并提取Y通道用于打分
        sr_y = cv2.cvtColor(sr_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        gt_y = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        
        mse = np.mean((sr_y - gt_y) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    except Exception as e:
        print(f"PSNR计算异常: {e}")
        return -1.0