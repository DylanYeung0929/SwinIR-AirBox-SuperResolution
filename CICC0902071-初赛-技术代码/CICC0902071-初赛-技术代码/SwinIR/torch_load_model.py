import torch
from network_swinir import SwinIR as net

# 初始化模型（参数需与训练设置一致）
model = net(upscale=4, in_chans=3, img_size=48, window_size=8,
            img_range=1., depths=[6,6,6,6], embed_dim=60, num_heads=[6,6,6,6],
            mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')

# 加载预训练权重
checkpoint = torch.load('SwinIR_4x.pth')
if 'params' in checkpoint:
    model.load_state_dict(checkpoint['params'])
elif 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint)
model.eval()  # 设置为评估模式