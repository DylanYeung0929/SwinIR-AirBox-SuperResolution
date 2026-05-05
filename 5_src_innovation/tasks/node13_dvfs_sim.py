import time
import os
import random

class DVFSSimulator:
    def __init__(self):
        print("[Node 13] 启动 DVFS (动态电压频率调整) 能耗仿真模型...")
        # 设定 BM1684X 模拟档位 (频率 MHz, 电压 V, 相对功耗 P = V^2 * f, 相对延迟 L = 1/f)
        # 归一化基准: Turbo档为 1.0
        self.gears = {
            'TURBO':  {'freq': 1000, 'volt': 0.90, 'pwr': 1.00, 'lat': 1.00},
            'NORMAL': {'freq': 750,  'volt': 0.80, 'pwr': 0.59, 'lat': 1.33},
            'ECO':    {'freq': 500,  'volt': 0.70, 'pwr': 0.30, 'lat': 2.00},
            'SLEEP':  {'freq': 200,  'volt': 0.55, 'pwr': 0.07, 'lat': 5.00}
        }
        
    def determine_gear(self, entropy_var):
        """根据图像信息熵(复杂度)决定硬件运行档位"""
        if entropy_var > 1500:
            return 'TURBO'
        elif entropy_var > 800:
            return 'NORMAL'
        elif entropy_var > 300:
            return 'ECO'
        else:
            return 'SLEEP'

    def simulate_video_stream(self, frames_entropy):
        """模拟连续视频流的功耗表现"""
        total_baseline_pwr = 0
        total_dvfs_pwr = 0
        
        gear_stats = {'TURBO': 0, 'NORMAL': 0, 'ECO': 0, 'SLEEP': 0}
        
        print("\n--- 视频流能耗动态调度仿真开始 ---")
        print(f"{'Frame':<8} | {'Entropy':<10} | {'Hardware Gear':<12} | {'Power Consumed':<15}")
        print("-" * 55)
        
        for i, var in enumerate(frames_entropy):
            # 基准方案：永远跑在最高频
            total_baseline_pwr += self.gears['TURBO']['pwr']
            
            # DVFS 动态方案
            gear_name = self.determine_gear(var)
            gear_stats[gear_name] += 1
            current_pwr = self.gears[gear_name]['pwr']
            total_dvfs_pwr += current_pwr
            
            if i < 10 or i == len(frames_entropy) - 1: # 仅打印部分日志避免刷屏
                print(f"#{i+1:<7} | {var:<10.1f} | {gear_name:<12} | {current_pwr*100:>5.1f}%")
            elif i == 10:
                print("... (中间帧日志省略) ...")

        avg_power_saving = (1 - total_dvfs_pwr / total_baseline_pwr) * 100
        
        print("\n==================================================")
        print("DVFS 仿真结果统计 (Simulation Summary)")
        print("==================================================")
        print(f"总处理帧数: {len(frames_entropy)} 帧")
        print(f"档位分布: TURBO({gear_stats['TURBO']}) | NORMAL({gear_stats['NORMAL']}) | ECO({gear_stats['ECO']}) | SLEEP({gear_stats['SLEEP']})")
        print(f"理论总体功耗节省 (Power Saving): {avg_power_saving:.2f}%")
        print("==================================================")

def run_node13_test():
    simulator = DVFSSimulator()
    
    # 构造一个模拟视频流的熵值分布
    # 场景假设: 无人机巡检
    # 前30帧: 高空平飞，基本是蓝天白云 (低熵)
    # 中间40帧: 靠近高压电塔，纹理极其复杂 (高熵)
    # 后30帧: 悬停监控地面单一平整路面 (中低熵)
    
    random.seed(42)
    stream_entropy = []
    
    # 使用原生 random 库替代 numpy
    for _ in range(30):
        stream_entropy.append(random.gauss(200, 50))   # 蓝天
    for _ in range(40):
        stream_entropy.append(random.gauss(1800, 200)) # 电塔
    for _ in range(30):
        stream_entropy.append(random.gauss(600, 100))  # 路面
    
    # 限制负数值
    stream_entropy = [max(0, val) for val in stream_entropy]
    
    simulator.simulate_video_stream(stream_entropy)

if __name__ == "__main__":
    run_node13_test()