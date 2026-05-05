import matplotlib.pyplot as plt
import numpy as np
import os
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def set_premium_style():
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = '#FFFFFF' 
    plt.rcParams['axes.facecolor'] = '#F8FAFC'   
    plt.rcParams['axes.edgecolor'] = '#CBD5E1'   
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#E2E8F0'       
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif', 'serif']

set_premium_style()

COLORS = {
    'cache_hit': '#1E40AF',
    'fps_boost': '#10B981',
    'text': '#0F172A',
    'annotation': '#64748B'
}

def plot_node11_temporal_efficiency():
    scenes = [
        'Static Scenery\n(Mining/Fixed)', 
        'Slow Dynamic\n(Urban/V2X)', 
        'Fast Motion\n(Highway)', 
        'Traditional Pipeline\n(Naive SR)'
    ]
    
    # 填入 AirBox 实机真实点火数据
    cache_hit = [75.0, 50.0, 15.0, 0.0] 
    fps_boost = [4.00, 1.85, 1.05, 1.0]

    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)

    x = np.arange(len(scenes))
    bars = ax1.bar(x, cache_hit, color=COLORS['cache_hit'], alpha=0.8, width=0.4, 
                   edgecolor='#1D4ED8', linewidth=1.5, label='VRAM Cache Hit Rate (%)')
    
    ax1.set_ylabel('VRAM Cache Hit Rate (%)', color=COLORS['cache_hit'], fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.tick_params(axis='y', labelcolor=COLORS['cache_hit'])

    ax2 = ax1.twinx()
    ax2.plot(x, fps_boost, color=COLORS['fps_boost'], marker='o', markersize=12, 
                     linewidth=4, label='FPS Boost Factor (x)', 
                     markeredgecolor='white', markeredgewidth=2)
    
    ax2.set_ylabel('Throughput Gain (Speedup Factor)', color=COLORS['fps_boost'], fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 5.0)
    ax2.tick_params(axis='y', labelcolor=COLORS['fps_boost'])

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax1.text(i, h - 8 if h > 10 else h + 2, f'{h:.1f}%', 
                 ha='center', va='bottom', color='white' if h > 10 else COLORS['cache_hit'], 
                 fontweight='heavy', fontsize=11)
        
        ax2.annotate(f'{fps_boost[i]:.2f}x', 
                     xy=(i, fps_boost[i]), 
                     xytext=(0, 15), 
                     textcoords="offset points",
                     ha='center', va='bottom', 
                     color=COLORS['fps_boost'], 
                     fontweight='black', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS['fps_boost'], alpha=0.8))

    plt.title('Node 11: Hardware-Aware Temporal Sparse Reuse Optimization', 
              fontsize=18, fontweight='bold', pad=30, color=COLORS['text'])
    
    ax1.annotate('Hardware Memory Hierarchy Benefit:\nComputation Skipped via VRAM Reuse', 
                 xy=(0, 75.0), xytext=(0.8, 95),
                 arrowprops=dict(arrowstyle="->", color=COLORS['annotation'], connectionstyle="arc3,rad=.2"),
                 fontsize=10, fontstyle='italic', color=COLORS['annotation'])

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenes, fontsize=11, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
               bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, facecolor='#FFFFFF')

    save_dir = "/home/dylanyeung/SwinIR_AirBox_Project/5_src_innovation/archive_report_charts"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "13_Node11_Temporal_Reuse_Efficiency.png")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"图表已生成至: {save_path}")

if __name__ == "__main__":
    plot_node11_temporal_efficiency()