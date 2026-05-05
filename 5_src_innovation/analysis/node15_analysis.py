import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# 屏蔽 Matplotlib 字体警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def set_premium_style():
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = '#FFFFFF' 
    plt.rcParams['axes.facecolor'] = '#F8FAFC'   
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'   
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#E2E8F0'       
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

set_premium_style()

COLORS = {
    'teacher': '#1E3A8A',   # Navy Blue
    'student': '#10B981',   # Emerald
    'gain': '#F59E0B',      # Amber
    'text': '#1E293B'
}

def plot_node15_distillation_report():
    # 1. 硬件资源消耗对比数据 (根据 AirBox 实测数据填充)
    metrics = ['VRAM Occupancy (MB)', 'Inference Latency (ms)', 'Model Size (MB)']
    teacher_vals = [940, 310, 582]
    student_vals = [120, 45, 85]  # 学生模型数据 (混合精度/瘦身版)

    # 2. 画质保持验证 (PSNR)
    scenes = ['Normal', 'Smooth', 'Complex', 'Moving']
    psnr_teacher = [33.6, 34.2, 32.8, 32.1]
    psnr_student = [31.5, 33.1, 30.2, 29.8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # 子图 A: 硬件资源瘦身效果
    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width/2, teacher_vals, width, label='Teacher (Full BF16)', color=COLORS['teacher'], alpha=0.8)
    ax1.bar(x + width/2, student_vals, width, label='Student (Mixed/Light)', color=COLORS['student'], alpha=0.8)
    
    ax1.set_yscale('log') # 对数坐标更清晰
    ax1.set_ylabel('Consumption (Log Scale)', fontweight='bold')
    ax1.set_title('A: Hardware Resource Efficiency (7x Faster)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()

    # 子图 B: 画质损失控制
    ax2.plot(scenes, psnr_teacher, color=COLORS['teacher'], marker='o', linewidth=3, label='Teacher Quality')
    ax2.plot(scenes, psnr_student, color=COLORS['student'], marker='s', linewidth=3, label='Student Quality')
    ax2.fill_between(scenes, psnr_student, psnr_teacher, color=COLORS['gain'], alpha=0.1, label='Quality Gap')
    
    ax2.set_ylabel('PSNR (dB)', fontweight='bold')
    ax2.set_title('B: Distillation Quality Loss Control', fontsize=14, fontweight='bold')
    ax2.set_ylim(25, 40)
    ax2.legend()

    plt.suptitle('Node 15: Knowledge Distillation & Dynamic Model Routing', fontsize=18, fontweight='black', y=1.02)
    
    # 保存路径
    save_dir = "/home/dylanyeung/SwinIR_AirBox_Project/5_src_innovation/archive_report_charts"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "18_Node15_Distillation_Analysis.png")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Node 15 知识蒸馏战报图表已生成: {save_path}")

if __name__ == "__main__":
    plot_node15_distillation_report()