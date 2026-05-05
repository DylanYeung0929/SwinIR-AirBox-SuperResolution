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
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']

set_premium_style()

COLORS = {
    'cpu': '#EF4444',     
    'tpu': '#10B981',     
    'text': '#0F172A',
    'accent': '#3B82F6'
}

def plot_node12_io_wall():
    labels = ['CPU (Numpy/Python)', 'Native C++ (Zero-Copy)']
    times_ms = [15.81, 3.94]
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    bars = ax1.bar(labels, times_ms, color=[COLORS['cpu'], COLORS['tpu']], width=0.4, edgecolor='#1E293B', linewidth=1.5)
    
    ax1.set_ylabel('Post-processing Latency (ms / Tile)', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax1.set_ylim(0, 20)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.2f} ms', 
                 ha='center', va='bottom', color=COLORS['text'], fontweight='black', fontsize=12)

    speedup = times_ms[0] / times_ms[1]
    ax1.annotate(f'Speedup: {speedup:.2f}x\nMemory Allocation Overhead Bypassed', 
                 xy=(1, times_ms[1] * 1.5), xytext=(0.5, times_ms[0] * 0.8),
                 arrowprops=dict(facecolor=COLORS['accent'], shrink=0.05, width=2, headwidth=10, connectionstyle="arc3,rad=-0.2"),
                 ha='center', va='center', color=COLORS['accent'], fontweight='black', fontsize=13,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#EFF6FF", ec=COLORS['accent'], alpha=0.9))

    plt.title('Node 12: Eradicating the Python GIL & I/O Memory Wall', 
              fontsize=16, fontweight='bold', pad=20, color=COLORS['text'])
    
    save_dir = "/home/dylanyeung/SwinIR_AirBox_Project/5_src_innovation/archive_report_charts"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "15_Node12_IO_Wall_Speedup.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Node 12 report generated successfully at: {save_path}")

if __name__ == "__main__":
    plot_node12_io_wall()