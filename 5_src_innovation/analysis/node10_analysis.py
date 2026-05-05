import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# 屏蔽警告
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

# 战报配色体系 (包含所有分级策略颜色)
COLORS = {
    'bw': '#6366F1',           # 基础靛蓝 (Native 4K)
    'psnr': '#F43F5E',         # 画质品红
    'text': '#1E293B',         # 文字深灰
    'collapse': '#94A3B8',     # 崩塌灰
    'success': '#059669',      # 甜点绿
    'danger': '#E11D48',       # 危险红
    'high_fidelity': '#0EA5E9' # 高保真天蓝
}

def plot_node10_report():
    # 🌟 核心重构数据：汇集我们在 AirBox 上的 3 次真实物理测试！
    scenarios = [
        'Native 4K\n(Uncompressed)', 
        'High-Fidelity\n(Q=120)', 
        'Sweet Spot\n(Q=50)', 
        'Extreme\n(Q=20)'
    ]
    # 真实带宽占用率 = 100% - 节省率
    bandwidth_usage = [100.0, 15.02, 2.70, 1.51]  
    # 对应画质评估 (极端压缩下会导致严重的方块伪影，PSNR暴跌)
    psnr_values = [40.0, 38.5, 31.5, 16.2]       

    fig, ax1 = plt.subplots(figsize=(12, 6.5), dpi=300)

    # 1. 带宽需求对比 (柱状图)
    bar_colors = [COLORS['bw'], COLORS['high_fidelity'], COLORS['success'], COLORS['collapse']]
    bars = ax1.bar(scenarios, bandwidth_usage, color=bar_colors, alpha=0.85, width=0.45, edgecolor='white', linewidth=1.5, label='Bandwidth Usage (%)')
    ax1.set_ylabel('Network Bandwidth Requirement (%)', color=COLORS['bw'], fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 125)
    ax1.tick_params(axis='y', labelcolor=COLORS['bw'])

    # 柱状图顶部数值标注
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h}%', ha='center', fontweight='bold', color=COLORS['text'], fontsize=10)

    # 2. 画质表现 (折线图)
    ax2 = ax1.twinx()
    ax2.plot(scenarios, psnr_values, color=COLORS['psnr'], marker='D', markersize=10, linewidth=3, label='TPU Reconstructed Quality (PSNR)', markeredgecolor='white', markeredgewidth=2)
    ax2.set_ylabel('Image Quality (PSNR dB)', color=COLORS['psnr'], fontsize=12, fontweight='bold')
    ax2.set_ylim(10, 45)
    ax2.tick_params(axis='y', labelcolor=COLORS['psnr'])

    # 折线图数值标注
    for i, v in enumerate(psnr_values):
        ax2.annotate(f'{v} dB', (i, psnr_values[i]), textcoords="offset points", xytext=(0, 15), ha='center', color=COLORS['psnr'], fontweight='bold')

    # 3. 装饰与大标题
    plt.title('Node 10: Information-Theoretic Limits & Tiered Cloud-Edge SR Strategy', fontsize=15, fontweight='bold', pad=25, color=COLORS['text'])
    
    # 🌟 标注1：无损高保真区 (Q=120)
    ax1.annotate('High-Fidelity Mode\n84.98% Bandwidth Saved!', xy=(1, 15.02), xytext=(0.3, 90),
                 arrowprops=dict(facecolor=COLORS['high_fidelity'], shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, fontweight='bold', color=COLORS['high_fidelity'], bbox=dict(boxstyle="round,pad=0.5", fc="#F0F9FF", ec=COLORS['high_fidelity'], alpha=0.9))

    # 🌟 标注2：工业甜点区 (Q=50)
    ax1.annotate('Optimal Sweet Spot\n97.30% Bandwidth Saved!', xy=(2, 2.70), xytext=(1.5, 60),
                 arrowprops=dict(facecolor=COLORS['success'], shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, fontweight='bold', color=COLORS['success'], bbox=dict(boxstyle="round,pad=0.5", fc="#F0FDF4", ec=COLORS['success'], alpha=0.9))

    # 🌟 标注3：语义崩塌区 (Q=20)
    ax1.annotate('Semantic Collapse\n(Severe Checkerboard)', xy=(3, 1.51), xytext=(2.6, 30),
                 arrowprops=dict(facecolor=COLORS['danger'], shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', color=COLORS['danger'], bbox=dict(boxstyle="round,pad=0.5", fc="#FFF1F2", ec=COLORS['danger'], alpha=0.9))

    # 合并图例 (放在中间偏上，避免遮挡折线)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1')

    plt.tight_layout()
    # 统一保存到总的图表档案库
    save_path = "../archive_report_charts/12_Node10_Bandwidth_Efficiency.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"📊 第 12 号全景战报(含高保真/甜点/崩塌三大物理指标)已归档: {save_path}")

if __name__ == "__main__":
    plot_node10_report()