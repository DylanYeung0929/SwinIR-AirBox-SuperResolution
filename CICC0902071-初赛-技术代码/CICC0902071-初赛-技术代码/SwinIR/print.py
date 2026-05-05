# 查看权重参数结构
checkpoint = torch.load('SwinIR_4x.pth')
print(checkpoint.keys())  # 查看参数层级结构
print(checkpoint['params'].keys())  # 具体参数名