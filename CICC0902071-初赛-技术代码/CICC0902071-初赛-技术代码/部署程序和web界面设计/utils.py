import numpy as np
import sophon.sail as sail
import cv2
import traceback

def preprocess(img_bm, bmcv, input_range='0_1', model_input_shape=None):
    try:
        print(f"原始图像宽度: {img_bm.width()}, 高度: {img_bm.height()}")
        
        # 获取目标尺寸
        if model_input_shape is None:
            target_size = (192, 192)
        else:
            # 模型输入形状: [N, C, H, W]
            target_size = (model_input_shape[3], model_input_shape[2])  # (H, W)
        
        print(f"目标尺寸: {target_size}")
        
        # 使用vpp_resize缩放图像
        resized_bm = sail.BMImage()
        if img_bm.width() != target_size[0] or img_bm.height() != target_size[1]:
            print(f"缩放图像: ({img_bm.width()}, {img_bm.height()}) -> {target_size}")
            bmcv.vpp_resize(img_bm, resized_bm, target_size[0], target_size[1])
        else:
            resized_bm = img_bm
            print("图像尺寸匹配, 无需缩放")
            
        # 颜色空间转换 BGR->RGB
        print("转换颜色空间 BGR->RGB")
        rgb_bm = bmcv.convert_format(resized_bm, sail.Format.FORMAT_RGB_PLANAR)
        
        # 转换为张量并归一化
        print("转换为张量")
        tensor = bmcv.bm_image_to_tensor(rgb_bm)
        img_data = tensor.asnumpy()
        print(f"张量形状: {img_data.shape}")
        
        # 归一化
        if input_range == '0_1':
            img_normalized = img_data.astype(np.float32) / 255.0
            print("归一化: 0-1范围")
        else:
            img_normalized = (img_data.astype(np.float32) * 2.0 / 255.0) - 1.0
            print("归一化: -1到1范围")
        

        # 确保张量是4维 (N, C, H, W)
        if len(img_normalized.shape) == 3:
            # 如果是3维 (C, H, W)，添加批次维度
            img_normalized = np.expand_dims(img_normalized, axis=0)
            print(f"添加批次维度后形状: {img_normalized.shape}")
        elif len(img_normalized.shape) == 4:
            # 已经是4维，无需修改
            print("张量已经是4维格式")
        else:
            print(f"不支持的张量形状: {img_normalized.shape}")
        
        return img_normalized.astype(np.float32)


    except Exception as e:
        print(f"预处理异常: {e}")
        traceback.print_exc()
        return None

# def postprocess(output_np, bmcv, input_range='0_1', model_output_shape=None):
#     try:
#         print(f"后处理输入形状: {output_np.shape}")
        
#         # 处理输出形状 - 转换为HWC格式 (高度, 宽度, 通道)
#         if len(output_np.shape) == 4:
#             # 去除批次维度
#             output_np = output_np[0].transpose(1, 2, 0)  # CHW -> HWC
#             print(f"转换后形状 (HWC): {output_np.shape}")
#         elif len(output_np.shape) == 3:
#             # 已经是3维，转换为HWC
#             if output_np.shape[0] == 3:  # CHW格式
#                 output_np = output_np.transpose(1, 2, 0)  # CHW -> HWC
#                 print(f"转换CHW->HWC后形状: {output_np.shape}")
#             elif output_np.shape[2] == 3:  # HWC格式
#                 print("已经是HWC格式")
#             else:
#                 print(f"无法处理的形状: {output_np.shape}")
#                 return None
        
#         # 还原范围
#         if input_range == '0_1':
#             img_data = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
#             print("还原范围: 0-255")
#         else:
#             img_data = np.clip((output_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
#             print("还原范围: -1到1 -> 0-255")
#         print(f"最终图像形状: {img_data.shape}")


#         # 创建Tensor对象
#         handle = bmcv.get_handle()  # 获取bmcv关联的handle
#         tensor = sail.Tensor(handle, img_data)
        

        
#         # 正确调用 tensor_to_bm_image
#         # 明确指定格式为RGB_PLANAR
#         print("将Tensor转换为BMImage (RGB_PLANAR)")
#         result_bm = bmcv.tensor_to_bm_image(tensor, bgr2rgb=False)

#         if result_bm is None:
#             print("tensor_to_bm_image失败")
#             return None
        
#         # 转换回BGR格式
#         print("转换颜色空间 RGB->BGR")
#         bgr_bm = bmcv.convert_format(result_bm, sail.Format.FORMAT_BGR_PACKED)
        
#         return bgr_bm
#     except Exception as e:
#         print(f"后处理异常: {e}")
#         traceback.print_exc()
#         return None

def postprocess(output_np, bmcv, input_range='0_1', model_output_shape=None):
    try:
        print(f"后处理输入形状: {output_np.shape}")
        
        # 处理输出形状 - 转换为HWC格式 (高度, 宽度, 通道)
        if len(output_np.shape) == 4:
            # 去除批次维度并转换为HWC
            output_np = output_np[0].transpose(1, 2, 0)  # CHW -> HWC
            print(f"转换后形状 (HWC): {output_np.shape}")
        elif len(output_np.shape) == 3:
            # 已经是3维，转换为HWC
            if output_np.shape[0] == 3:  # CHW格式
                output_np = output_np.transpose(1, 2, 0)  # CHW -> HWC
                print(f"转换CHW->HWC后形状: {output_np.shape}")
            elif output_np.shape[2] == 3:  # HWC格式
                print("已经是HWC格式")
            else:
                print(f"无法处理的形状: {output_np.shape}")
                return None
        
        # 还原范围
        if input_range == '0_1':
            img_data = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
            print("还原范围: 0-255")
        else:
            img_data = np.clip((output_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
            print("还原范围: -1到1 -> 0-255")
        
        print(f"最终图像形状: {img_data.shape}")
        
        # 直接使用OpenCV进行颜色空间转换
        # 模型输出是RGB格式，转换为OpenCV的BGR格式
        if img_data.shape[2] == 3:  # RGB格式
            bgr_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            print("转换颜色空间 RGB->BGR (使用OpenCV)")
        else:
            bgr_img = img_data
        
        return bgr_img  # 直接返回OpenCV格式的numpy数组
        
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
            print(f"调整GT尺寸: {gt_img.shape} -> {sr_img.shape}")
            gt_img = cv2.resize(gt_img, (sr_img.shape[1], sr_img.shape[0]))
            print(f"调整GT尺寸为: {sr_img.shape}")
            
        # 转换为YCrCb并提取Y通道
        sr_y = cv2.cvtColor(sr_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        gt_y = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        
        # 计算MSE
        mse = np.mean((sr_y - gt_y) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    except Exception as e:
        print(f"PSNR计算异常: {e}")
        return -1.0