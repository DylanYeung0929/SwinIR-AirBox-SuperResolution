import os
import time
from pipeline import Pipeline
from config import get_args

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 收集输入图像 (添加格式检查)
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入路径不存在 {args.input_dir}")
        return

    # 收集输入图像
    if os.path.isfile(args.input_dir):
        input_paths = [args.input_dir]
    elif os.path.isdir(args.input_dir):
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        input_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                if os.path.splitext(f)[1].lower() in exts]
    else:
        print("无效输入路径")
        return

    print(f"找到 {len(input_paths)} 个图像")

    # 创建处理管道
    pipeline = Pipeline(args)
    
    print(f"启动 {args.num_streams} 个并行流...")
    start_time = time.time()
    results = pipeline.start(input_paths)
    total_time = time.time() - start_time
    
    # 打印统计信息
    print(f"\n所有任务完成! 总耗时: {total_time:.2f}秒")
    print(f"共处理 {len(results)} 张图像")
    
    if args.test_y:
        psnr_values = [r['psnr'] for r in results.values()]
        if psnr_values:
            avg_psnr = sum(psnr_values) / len(psnr_values)
        else:
            avg_psnr = 0.0
        print(f"平均PSNR: {avg_psnr:.2f}dB")
    
    # 打印每个图像的结果
    for path, result in results.items():
        psnr_info = f", PSNR={result['psnr']:.2f}dB" if args.test_y else ""
        print(f"{os.path.basename(path)}: 耗时={result['time']:.2f}s{psnr_info}")
        
if __name__ == '__main__':
    main()
