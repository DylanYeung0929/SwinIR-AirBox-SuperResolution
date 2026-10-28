import argparse

def get_args():
    parser = argparse.ArgumentParser(description="SwinIR Sophon TPU 多流推理")
    parser.add_argument('--input_dir', type=str, required=True, help="输入图像文件或目录")
    parser.add_argument('--bmodel_path', type=str, required=True, help="SwinIR BModel 文件路径")
    parser.add_argument('--device_id', type=int, default=0, help="TPU 设备 ID")
    parser.add_argument('--save_dir', type=str, default='./results', help="结果保存目录")
    parser.add_argument('--scale', type=int, required=True, help="超分倍数")
    parser.add_argument('--window_size', type=int, required=True, help="模型滑窗大小")
    parser.add_argument('--tile', type=int, default=192, help="分块推理大小")
    parser.add_argument('--tile_overlap', type=int, default=64, help="分块重叠大小")
    parser.add_argument('--input_range', choices=['0_1','-1_1'], default='0_1', help="输入归一化范围")
    parser.add_argument('--tta', action='store_true', help="启用TTA")
    parser.add_argument('--num_streams', type=int, default=4, help="并行流数量")
    parser.add_argument('--test_y', action='store_true', help='测试 Y 通道')
    return parser.parse_args()
