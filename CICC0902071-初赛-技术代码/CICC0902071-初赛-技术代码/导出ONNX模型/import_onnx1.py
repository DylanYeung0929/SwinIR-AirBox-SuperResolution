import torch
import torch.nn.functional as F
from network_swinir import SwinIR

def convert_swinir_to_onnx(pth_path, onnx_path, scale=4):
    # 初始化模型
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
    
    # 加载权重
    checkpoint = torch.load(pth_path, map_location='cpu')
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    else:
        state_dict = checkpoint
    
    # 处理权重键名
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'attn_mask' in k:
            continue
        new_k = k.replace('module.', '')  # 移除分布式训练的前缀
        new_state_dict[new_k] = v
    
    # 加载权重
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Weights loaded with messages: {msg}")
    model.eval()

    # 改进的尺寸调整函数（避免Python整数操作）
    class PadInput(torch.nn.Module):
        def forward(self, x):
            h, w = x.shape[2:]
            mod_pad_h = (8 - h % 8) % 8
            mod_pad_w = (8 - w % 8) % 8
            return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    
    # 创建模型包装器处理动态尺寸
    class SwinIRWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.pad = PadInput()
            
        def forward(self, x):
            # 先处理填充，然后模型前向
            x = self.pad(x)
            return self.model(x)
    
    # 创建包装后的模型
    wrapped_model = SwinIRWrapper(model)
    wrapped_model.eval()

    # 创建动态输入（用于测试）
    dummy_input = torch.randn(1, 3, 64, 64)

    # 导出ONNX - 使用包装后的模型
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        opset_version=17,  # 使用最新opset
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        },
        do_constant_folding=False,
        verbose=True
    )

if __name__ == "__main__":
    # 先安装ONNX（伪代码，实际需要在命令行执行）
    # !pip install onnx
    
    convert_swinir_to_onnx(
        pth_path="002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
        onnx_path="SwinIR_light_x4.onnx"
    )