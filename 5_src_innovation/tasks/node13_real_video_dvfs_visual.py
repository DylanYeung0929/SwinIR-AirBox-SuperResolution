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
        print("[Node 13] 启动 AirBox 实机 DVFS 可视化渲染引擎...")
        if has_sophon:
            print("[成功] 检测到 Sophon SDK，确认为 BM1684X 实机环境！")
        print("="*60)
        
        self.gears = {
            'TURBO':  {'freq': 1000, 'pwr': 1.00, 'color': (0, 0, 255)},   # Red
            'NORMAL': {'freq': 750,  'pwr': 0.59, 'color': (0, 165, 255)}, # Orange
            'ECO':    {'freq': 500,  'pwr': 0.30, 'color': (0, 255, 0)},   # Green
            'SLEEP':  {'freq': 200,  'pwr': 0.07, 'color': (255, 0, 0)}    # Blue
        }
        self.frame_count = 0
        self.calibration_frames = 30
        self.entropy_history = []
        self.th_turbo = 1500
        self.th_normal = 800
        self.th_eco = 300

    def calibrate_thresholds(self):
        mean_entropy = np.mean(self.entropy_history)
        std_entropy = np.std(self.entropy_history)
        self.th_turbo = mean_entropy + 1.5 * std_entropy
        self.th_normal = mean_entropy + 0.5 * std_entropy
        self.th_eco = mean_entropy - 0.5 * std_entropy
        print(f"[自适应标定完成] 基线: {mean_entropy:.2f} | TURBO > {self.th_turbo:.2f}")

    def determine_gear(self, entropy_var):
        if entropy_var > self.th_turbo: return 'TURBO'
        elif entropy_var > self.th_normal: return 'NORMAL'
        elif entropy_var > self.th_eco: return 'ECO'
        else: return 'SLEEP'

    def process_and_export_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"--- 正在分析并渲染视频: {output_path} ---")

        while True:
            ret, frame = cap.read()
            if not ret: break

            self.frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if self.frame_count <= self.calibration_frames:
                self.entropy_history.append(var)
                gear_name = 'CALIBRATING'
                color = (255, 255, 255)
                pwr_str = "---"
                if self.frame_count == self.calibration_frames:
                    self.calibrate_thresholds()
            else:
                gear_name = self.determine_gear(var)
                color = self.gears[gear_name]['color']
                pwr_str = f"{self.gears[gear_name]['pwr']*100:.1f}%"

            # 绘制高级 OSD (On-Screen Display)
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (380, 140), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            cv2.putText(frame, "BM1684X Real-time DVFS Monitor", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {self.frame_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Image Entropy: {var:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(frame, f"TPU Gear: {gear_name}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Power: {pwr_str}", (240, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            out.write(frame)

            if self.frame_count % 50 == 0:
                print(f"已处理 {self.frame_count} 帧...")

        cap.release()
        out.release()
        print(f"✅ 可视化视频已导出至: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help="输入视频路径")
    parser.add_argument('--output_path', type=str, default='node13_dvfs_demo.mp4', help="输出可视化视频路径")
    args = parser.parse_args()
    
    engine = DVFSRealEngine()
    engine.process_and_export_video(args.video_path, args.output_path)