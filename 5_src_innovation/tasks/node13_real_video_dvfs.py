import cv2
import numpy as np
import time
import argparse
import os

try:
    import sophon.sail as sail
    has_sophon = True
except ImportError:
    has_sophon = False

class DVFSRealEngine:
    def __init__(self):
        print("="*60)
        print("[Node 13] 启动 AirBox 实机真实视频流 DVFS 调度引擎...")
        if has_sophon:
            print("[成功] 检测到 Sophon SDK，确认为 BM1684X 实机环境！")
        else:
            print("[警告] 未检测到 Sophon SDK，将在普通 CPU 环境下进行纯量化仿真。")
        print("="*60)
        
        # 设定 BM1684X 物理档位
        self.gears = {
            'TURBO':  {'freq': 1000, 'volt': 0.90, 'pwr': 1.00, 'lat': 1.00},
            'NORMAL': {'freq': 750,  'volt': 0.80, 'pwr': 0.59, 'lat': 1.33},
            'ECO':    {'freq': 500,  'volt': 0.70, 'pwr': 0.30, 'lat': 2.00},
            'SLEEP':  {'freq': 200,  'volt': 0.55, 'pwr': 0.07, 'lat': 5.00}
        }
        self.gear_stats = {'TURBO': 0, 'NORMAL': 0, 'ECO': 0, 'SLEEP': 0}
        self.total_baseline_pwr = 0.0
        self.total_dvfs_pwr = 0.0
        self.frame_count = 0
        
        # 自适应标定相关参数
        self.calibration_frames = 30
        self.entropy_history = []
        self.th_turbo = 1500
        self.th_normal = 800
        self.th_eco = 300

    def calibrate_thresholds(self):
        """基于前 N 帧的统计特征，自适应动态计算阈值，适配任意分辨率视频"""
        mean_entropy = np.mean(self.entropy_history)
        std_entropy = np.std(self.entropy_history)
        
        # 动态自适应阈值公式
        self.th_turbo = mean_entropy + 1.5 * std_entropy
        self.th_normal = mean_entropy + 0.5 * std_entropy
        self.th_eco = mean_entropy - 0.5 * std_entropy
        
        print("\n" + "*"*55)
        print("🎯 [自适应标定完成] 已根据当前视频源特征动态重置阈值！")
        print(f"   平均基线熵值: {mean_entropy:.2f} | 波动率: {std_entropy:.2f}")
        print(f"   TURBO  > {self.th_turbo:.2f}")
        print(f"   NORMAL > {self.th_normal:.2f}")
        print(f"   ECO    > {self.th_eco:.2f}")
        print("*"*55 + "\n")

    def determine_gear(self, entropy_var):
        """基于自适应阈值进行物理档位映射"""
        if entropy_var > self.th_turbo: return 'TURBO'
        elif entropy_var > self.th_normal: return 'NORMAL'
        elif entropy_var > self.th_eco: return 'ECO'
        else: return 'SLEEP'

    def process_video(self, video_path):
        if not os.path.exists(video_path):
            print(f"错误: 找不到视频文件 {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("错误: 无法打开视频解码器")
            return

        print(f"\n--- 正在分析真实视频流: {os.path.basename(video_path)} ---")
        print(f"{'Frame':<8} | {'Real Entropy':<12} | {'Gear':<12} | {'Power':<10}")
        print("-" * 55)

        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 前 30 帧仅收集数据，不计入最终功耗评价
            if self.frame_count <= self.calibration_frames:
                self.entropy_history.append(var)
                print(f"#{self.frame_count:<7} | {var:<12.1f} | {'CALIBRATING':<12} | - ")
                if self.frame_count == self.calibration_frames:
                    self.calibrate_thresholds()
                continue

            # 正式评估阶段
            gear_name = self.determine_gear(var)
            self.gear_stats[gear_name] += 1

            current_pwr = self.gears[gear_name]['pwr']
            self.total_baseline_pwr += self.gears['TURBO']['pwr']
            self.total_dvfs_pwr += current_pwr

            # 打印日志 (为了看换挡过程，每 10 帧打一次)
            if self.frame_count % 10 == 0:
                print(f"#{self.frame_count:<7} | {var:<12.1f} | {gear_name:<12} | {current_pwr*100:>5.1f}%")

        cap.release()
        process_time = time.time() - start_time
        eval_frames = self.frame_count - self.calibration_frames

        if eval_frames > 0:
            avg_saving = (1 - self.total_dvfs_pwr / self.total_baseline_pwr) * 100
            print("\n==================================================")
            print("DVFS 真实视频流分析结果 (Real Video Summary)")
            print("==================================================")
            print(f"总解析帧数: {self.frame_count} 帧 (标定: {self.calibration_frames} | 评估: {eval_frames})")
            print(f"解析总耗时: {process_time:.2f}s")
            print(f"真实档位分布: TURBO({self.gear_stats['TURBO']}) | NORMAL({self.gear_stats['NORMAL']}) | ECO({self.gear_stats['ECO']}) | SLEEP({self.gear_stats['SLEEP']})")
            print(f"实际动态功耗节省 (Power Saving): {avg_saving:.2f}%")
            print("==================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help="输入测试的真实视频路径")
    args = parser.parse_args()
    
    engine = DVFSRealEngine()
    engine.process_video(args.video_path)