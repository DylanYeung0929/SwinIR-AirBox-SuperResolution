import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# 屏蔽 Matplotlib 找不到字体时的烦人警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ==========================================
# 🎨 苹果级高级学术白底风 (Premium Academic Light Theme)
# ==========================================
def set_premium_style():
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = '#FFFFFF' 
    plt.rcParams['axes.facecolor'] = '#F8FAFC'   
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'   
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#E2E8F0'       
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    plt.rcParams['text.color'] = '#1E293B'       
    plt.rcParams['axes.labelcolor'] = '#475569'  
    plt.rcParams['xtick.color'] = '#64748B'
    plt.rcParams['ytick.color'] = '#64748B'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif', 'serif']
    plt.rcParams['axes.titlepad'] = 20

set_premium_style()

COLORS = {
    'primary': '#2563EB',   # 科技蓝
    'danger': '#E11D48',    # 能量红
    'gray_dark': '#334155',
    'gray_light': '#94A3B8'
}

# ==========================================
# 📊 真实数据导入 (基于你之前运行的结果)
# ==========================================
thresholds = ['0\n(Full TTA)', '800', '2000\n(Sweet Spot)', '99999\n(None TTA)']
latency = [3904.95, 3754.10, 3124.77, 498.46]  # 毫秒
psnr = [38.80, 38.79, 38.69, 37.85]           # dB

def plot_node8_tta_analysis():
    # 【路径更新】：统一保存到总战报图表库
    save_dir = "../archive_report_charts"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
    
    # 1. 绘制耗时柱状图
    bars = ax1.bar(thresholds, latency, color=COLORS['primary'], alpha=0.8, width=0.45, edgecolor='white', linewidth=1.5, label='Inference Latency (ms)')
    ax1.set_xlabel('Decision Threshold (Laplacian Variance)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inference Latency (ms)', color=COLORS['primary'], fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax1.set_ylim(0, 5000)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 100, f'{int(height)}ms', 
                 ha='center', va='bottom', color=COLORS['primary'], fontweight='bold', fontsize=10)

    # 2. 绘制 PSNR 折线图
    ax2 = ax1.twinx()
    ax2.plot(thresholds, psnr, color=COLORS['danger'], marker='D', markersize=10, linewidth=3, label='Quality (PSNR)', markeredgecolor='white', markeredgewidth=2)
    ax2.fill_between(thresholds, psnr, 37, color=COLORS['danger'], alpha=0.05)
    ax2.set_ylabel('Image Quality (PSNR dB)', color=COLORS['danger'], fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COLORS['danger'])
    ax2.set_ylim(37, 39.5)

    for i, v in enumerate(psnr):
        ax2.annotate(f'{v} dB', (i, psnr[i]), textcoords="offset points", xytext=(0, 15), 
                     ha='center', color=COLORS['danger'], fontweight='bold', fontsize=10)

    plt.title('Node 8: Dynamic TTA Strategy Efficiency Analysis', fontsize=16, fontweight='bold', color=COLORS['gray_dark'])
    
    # 3. 保存为编号 11 的正式战报图表
    save_path = os.path.join(save_dir, "11_Node8_TTA_Efficiency_Analysis.png")
    plt.savefig(save_path)
    print(f"📊 Node 8 分析图表已归档至总库: {save_path}")

if __name__ == "__main__":
    plot_node8_tta_analysis()