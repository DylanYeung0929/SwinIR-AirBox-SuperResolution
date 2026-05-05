import cProfile
import pstats
import io
import os

def run_profiling():
    # 动态导入我们刚才的部署引擎
    try:
        from node11_real_deploy import run_node11_test
        import argparse
    except ImportError:
        print("❌ 找不到 node11_real_deploy.py，请确保在同一目录下运行。")
        return

    # 设置模拟参数
    class Args:
        img_path = './detection_test.jpg'
        bmodel_path = './SwinIR_1684x_bf16.bmodel'
        threshold = 10.0
        device_id = 0
        tile = 192
        scale = 4
        window_size = 8
        input_range = '0_1'

    print("🕵️ 开始进行系统级性能溯源 (Profiling)...")
    
    pr = cProfile.Profile()
    pr.enable()
    
    # 运行测试逻辑
    run_node11_test(Args())
    
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative' # 按累计耗时排序
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    
    print("\n" + "="*60)
    print("📊 Node 12: 系统函数耗时排行榜 (Top 10)")
    print("="*60)
    # 打印前 15 行最耗时的函数调用
    ps.print_stats(15)
    print(s.getvalue())

if __name__ == "__main__":
    run_profiling()