import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# 屏蔽 Matplotlib 找不到字体时的警告
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
    'noise': '#3B82F6',     # Blue
    'blocking': '#F59E0B',  # Amber
    'total': '#E11D48',     # Red
    'safe': '#10B981',      # Emerald
    'text': '#1E293B'
}

def plot_node14_blind_sr_report():
    # 模拟不同退化场景下的感知数据
    scenarios = ['Clean Source', 'Mild Compression', 'UAV Weak Signal (Real Test)', 'Severe Artifacts']
    
    # 基于实测数据：Noise(16.26), Block(13.21), Total(39.53)
    noise_scores = [2.5, 8.2, 16.26, 25.4]
    block_scores = [1.1, 4.5, 13.21, 22.8]
    # 综合指数公式: (Noise * 0.4) + (Block * 2.5)
    total_index = [(n * 0.4 + b * 2.5) for n, b in zip(noise_scores, block_scores)]
    
    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # 绘制堆叠/分组指标
    rects1 = ax.bar(x - width, noise_scores, width, label='Noise Response', color=COLORS['noise'], alpha=0.7)
    rects2 = ax.bar(x, block_scores, width, label='JPEG Blocking Score', color=COLORS['blocking'], alpha=0.7)
    
    # 绘制综合退化指数折线
    line = ax.plot(x, total_index, color=COLORS['total'], marker='o', markersize=10, 
                   linewidth=4, label='Integrated Degradation Index', markeredgecolor='white', markeredgewidth=2)

    # 绘制警戒线 (Threshold = 30.0)
    ax.axhline(y=30, color=COLORS['total'], linestyle='--', linewidth=2, alpha=0.6)
    ax.text(-0.4, 31, 'Defense Trigger Threshold (30.0)', color=COLORS['total'], fontweight='bold', fontsize=10)

    # 标注实测点
    ax.annotate('AirBox Real Test:\nAction Triggered!', xy=(2, 39.53), xytext=(1.2, 55),
                 arrowprops=dict(facecolor=COLORS['text'], shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, fontweight='black', color=COLORS['text'],
                 bbox=dict(boxstyle="round,pad=0.5", fc="#FFF1F2", ec=COLORS['total'], alpha=0.9))

    # 性能标注
    plt.figtext(0.15, 0.82, "Perception Brain Latency: 0.385 ms", fontsize=14, 
                fontweight='black', color=COLORS['safe'], bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['safe']))

    ax.set_ylabel('Perception Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('Node 14: Adaptive Blind-SR Perception & Defense Mechanism', fontsize=16, fontweight='bold', pad=30)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 80)
    ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    
    # 保存至战报总库
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    save_dir = os.path.join(project_root, "archive_report_charts")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "17_Node14_BlindSR_Perception_Analysis.png")
    plt.savefig(save_path)
    print(f"Node 14 盲超分感知分析战报已生成: {save_path}")

if __name__ == "__main__":
    plot_node14_blind_sr_report()