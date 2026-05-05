import torch
import numpy as np
import onnxruntime as ort
from network_swinir import SwinIR  # 确保从您的原始代码中导入模型
import matplotlib.pyplot as plt
import time

def load_pytorch_model(pth_path):
    """加载PyTorch模型并返回"""
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        resi_connection='1conv',
        upsampler='pixelshuffledirect'
    )
    
    checkpoint = torch.load(pth_path, map_location='cpu')
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'attn_mask' in k:
            continue
        new_k = k.replace('module.', '')
        new_state_dict[new_k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def run_pytorch_inference(model, input_data):
    """在PyTorch模型上运行推理并计时"""
    start_time = time.time()
    with torch.no_grad():
        output = model(input_data)
    inference_time = time.time() - start_time
    return output, inference_time

def run_onnx_inference(onnx_path, input_data):
    """在ONNX模型上运行推理并计时"""
    # 准备ONNX运行会话
    ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # 将输入转换为ONNX所需的格式
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.cpu().numpy()
    
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    # 运行推理
    start_time = time.time()
    outputs = ort_session.run([output_name], {input_name: input_data})
    inference_time = time.time() - start_time
    
    return outputs[0], inference_time

def compare_models(pth_path, onnx_path, image_size=(64, 64)):
    """比较PyTorch和ONNX模型的输出差异"""
    # 1. 准备测试输入
    input_tensor = torch.rand(1, 3, *image_size)  # 随机输入
    print(f"Input shape: {input_tensor.shape}")
    
    # 2. 加载并运行PyTorch模型
    pytorch_model = load_pytorch_model(pth_path)
    pytorch_output, pt_time = run_pytorch_inference(pytorch_model, input_tensor.clone())
    
    # 3. 运行ONNX推理
    onnx_output, onnx_time = run_onnx_inference(onnx_path, input_tensor.clone())
    
    # 4. 将输出转换为相同格式
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.cpu().numpy()
    if isinstance(onnx_output, torch.Tensor):
        onnx_output = onnx_output.cpu().numpy()
    
    # 5. 计算差异指标
    abs_diff = np.abs(pytorch_output - onnx_output)
    mae = np.mean(abs_diff)  # 平均绝对误差
    max_diff = np.max(abs_diff)  # 最大绝对误差
    mse = np.mean((pytorch_output - onnx_output) ** 2)  # 均方误差
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))  # PSNR指标
    
    print("\n--- Comparison Results ---")
    print(f"PyTorch inference time: {pt_time:.4f} seconds")
    print(f"ONNX inference time: {onnx_time:.4f} seconds")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Max Absolute Difference: {max_diff:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.8f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
    
    # 6. 可视化差异
    plt.figure(figsize=(15, 10))
    
    # 原始输入图像
    plt.subplot(231)
    plt.title("Input Image")
    plt.imshow(np.clip(input_tensor[0].permute(1, 2, 0).numpy(), 0, 1))
    
    # PyTorch输出
    plt.subplot(232)
    plt.title("PyTorch Output")
    plt.imshow(np.clip(pytorch_output[0].transpose(1, 2, 0), 0, 1))
    
    # ONNX输出
    plt.subplot(233)
    plt.title("ONNX Output")
    plt.imshow(np.clip(onnx_output[0].transpose(1, 2, 0), 0, 1))
    
    # 差异图
    plt.subplot(234)
    plt.title("Absolute Difference (Scaled)")
    diff_plot = np.mean(abs_diff[0], axis=0)
    plt.imshow(diff_plot, cmap='hot', vmin=0, vmax=0.1)
    plt.colorbar()
    
    # 像素级差异分布
    plt.subplot(235)
    plt.title("Error Distribution")
    abs_diff_flat = abs_diff.flatten()
    plt.hist(abs_diff_flat, bins=100, range=(0, 0.1))
    plt.xlabel("Absolute Difference")
    plt.ylabel("Count")
    
    # 散点比较图
    plt.subplot(236)
    plt.title("Output Values Comparison")
    plt.scatter(pytorch_output.flatten(), onnx_output.flatten(), alpha=0.1)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("PyTorch Output")
    plt.ylabel("ONNX Output")
    
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()
    
    # 7. 检查显著性差异
    if mae > 0.001 or max_diff > 0.01:
        print("\n⚠️ WARNING: Significant differences detected! Possible conversion issues.")
    elif mae > 0.0001:
        print("\nℹ️ NOTE: Minor differences detected, likely from numerical precision.")
    else:
        print("\n✅ SUCCESS: Models match with high precision.")

if __name__ == "__main__":
    # 配置路径 - 替换为您的实际路径
    pth_path = "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
    onnx_path = "SwinIR_light_x4.onnx"
    
    # 运行比较
    compare_models(pth_path, onnx_path, image_size=(64, 64))
    
    # 可选的额外测试尺寸
    for size in [(48, 48), (64, 64), (128, 72), (96, 128)]:
        print(f"\nTesting with size: {size[0]}x{size[1]}")
        try:
            compare_models(pth_path, onnx_path, image_size=size)
        except Exception as e:
            print(f"❌ Error with size {size}: {str(e)}")