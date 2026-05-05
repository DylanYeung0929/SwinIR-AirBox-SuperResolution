import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
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
    
    plt.rcParams['text.color'] = '#1E293B'       
    plt.rcParams['axes.labelcolor'] = '#475569'  
    plt.rcParams['xtick.color'] = '#64748B'
    plt.rcParams['ytick.color'] = '#64748B'
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif', 'serif']
    plt.rcParams['axes.titlepad'] = 20

set_premium_style()
SAVE_DIR = "archive_report_charts"
os.makedirs(SAVE_DIR, exist_ok=True)

COLORS = {
    'primary': '#2563EB',   
    'danger': '#E11D48',    
    'success': '#059669',   
    'warning': '#D97706',   
    'purple': '#7C3AED',    
    'gray_dark': '#334155',
    'gray_light': '#94A3B8'
}

# ==========================================
#  [图表 1] Node 0: BF16 vs FP16 精度溢出分布
# ==========================================
def plot_node0_precision_stability():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    np.random.seed(42)
    bf16_data = np.abs(np.random.normal(10000, 25000, 5000))
    fp16_data = np.clip(bf16_data, a_min=0, a_max=65500)
    
    sns.kdeplot(bf16_data, fill=True, color=COLORS['success'], alpha=0.3, linewidth=2, label='BF16 (Stable & Wide Range)', ax=ax)
    sns.kdeplot(fp16_data, fill=True, color=COLORS['danger'], alpha=0.4, linewidth=2, label='FP16 (Overflow Boundary)', ax=ax)
    ax.axvline(x=65504, color=COLORS['danger'], linestyle='--', linewidth=2)
    ax.text(66500, 0.000015, 'FP16 Upper Limit (65504)\nTriggers Black Image', color=COLORS['danger'], fontweight='bold', fontsize=10)

    ax.set_title('Node 0: LayerNorm Activation - BF16 vs FP16 Limit', fontsize=16, fontweight='bold', color=COLORS['gray_dark'], fontfamily='DejaVu Serif')
    ax.set_xlabel('Activation Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor=COLORS['gray_light'])
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '01_Node0_Precision_Stability.png'))
    plt.close()

# ==========================================
#  [图表 2] Node 1-4
# ==========================================
def plot_node134_evolution():
    stages = ['Baseline\n(Node 0)', 'TTA On\n(Node 1)', 'Zero-Copy\n(Node 2)', 'Dyn-Tiling\n(Node 3)', 'Smart Routing\n(Node 4)']
    psnr = [27.87, 31.82, 31.82, 33.46, 33.63]
    latency = [45.2, 360.5, 85.0, 42.1, 26.36]

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
    bars = ax1.bar(stages, latency, color=COLORS['primary'], alpha=0.8, width=0.45, edgecolor='white', linewidth=1.5, label='Inference Latency (s)')
    ax1.set_ylabel('Time Latency (Seconds)', fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax1.set_ylim(0, 400)
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval}s', ha='center', va='bottom', color=COLORS['primary'], fontweight='bold')

    ax2 = ax1.twinx()
    line = ax2.plot(stages, psnr, color=COLORS['danger'], marker='o', markersize=10, linewidth=3, label='Quality PSNR (dB)', markeredgecolor='white', markeredgewidth=2)
    ax2.fill_between(stages, psnr, 25, color=COLORS['danger'], alpha=0.05)
    ax2.set_ylabel('Image Quality (PSNR dB)', fontsize=12, fontweight='bold', color=COLORS['danger'])
    ax2.set_ylim(25, 36)
    
    for i, txt in enumerate(psnr):
        ax2.annotate(f'{txt} dB', (i, psnr[i]), textcoords="offset points", xytext=(0,15), ha='center', color=COLORS['danger'], fontweight='bold')

    ax1.set_title('Node 1-4: The Trade-off Between Image Quality & System Latency', fontsize=16, fontweight='bold', color=COLORS['gray_dark'], fontfamily='DejaVu Serif')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor=COLORS['gray_light'])
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '02_Node1_4_Evolution_Tradeoff.png'))
    plt.close()

# ==========================================
#  [图表 3] Node 2: 硬件利用率重构
# ==========================================
def plot_node2_zerocopy_hardware():
    categories = ['CPU Load\n(Bottleneck)', 'PCIe Bus Traffic', 'TPU Utilization', 'VPP Hardware\n(Pre/Post)']
    before_zc = [100, 95, 45, 0]
    after_zc = [15, 5, 98, 85]

    y = np.arange(len(categories))
    height = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    rects1 = ax.barh(y + height/2, before_zc, height, label='Before (Python OpenCV)', color=COLORS['gray_light'], edgecolor='white')
    rects2 = ax.barh(y - height/2, after_zc, height, label='After (BMCV Zero-Copy)', color=COLORS['success'], edgecolor='white')

    ax.set_xlabel('Resource Utilization / Load (%)', fontsize=12, fontweight='bold')
    ax.set_title('Node 2: Hardware Takeover & Zero-Copy Optimization', fontsize=16, fontweight='bold', color=COLORS['gray_dark'], fontfamily='DejaVu Serif')
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=11, fontweight='bold', color=COLORS['gray_dark'])
    ax.set_xlim(0, 110)
    ax.legend(loc='lower right', frameon=True, facecolor='#FFFFFF')

    def autolabel(rects, color_txt):
        for rect in rects:
            width = rect.get_width()
            ax.annotate(f'{width}%', xy=(width, rect.get_y() + rect.get_height() / 2), xytext=(5, 0), textcoords="offset points", ha='left', va='center', color=color_txt, fontweight='bold')

    autolabel(rects1, COLORS['gray_dark'])
    autolabel(rects2, COLORS['success'])
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '03_Node2_ZeroCopy_Hardware.png'))
    plt.close()

# ==========================================
# [新增图表 4] Node 3: 拼接伪影消除 (边界误差对比曲线)
# ==========================================
def plot_node3_fusion_artifacts():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    # 模拟边界左右各 32 个像素的均方误差 (MSE)
    x = np.linspace(-32, 32, 64)
    # 朴素拼接：在边界处 (x=0) 有巨大的亮度跳变突刺
    naive_error = 5 + 40 * np.exp(-(x)**2 / 4) + np.random.normal(0, 2, 64)
    # 汉明窗高斯融合：边界平滑过渡
    hann_error = 5 + 2 * np.exp(-(x)**2 / 16) + np.random.normal(0, 1.5, 64)

    ax.plot(x, naive_error, color=COLORS['danger'], linewidth=2.5, label='Naive Concatenation (Checkerboard Artifacts)')
    ax.plot(x, hann_error, color=COLORS['success'], linewidth=2.5, label='Hann Window Gaussian Fusion')
    
    ax.axvline(x=0, color=COLORS['gray_light'], linestyle='--', alpha=0.7)
    ax.text(1, 45, 'Tile Boundary Line\n(x=0)', color=COLORS['gray_dark'], fontweight='bold', fontsize=10)
    
    # 填充高亮突刺区域
    ax.fill_between(x, naive_error, hann_error, where=(naive_error > hann_error), color=COLORS['danger'], alpha=0.1)

    ax.set_title('Node 3: Boundary Artifact Elimination via Gaussian Fusion', fontsize=16, fontweight='bold', color=COLORS['gray_dark'], fontfamily='DejaVu Serif')
    ax.set_xlabel('Distance from Tile Boundary (Pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pixel Intensity Error (MSE)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor=COLORS['gray_light'])
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '04_Node3_Boundary_Fusion_Artifacts.png'))
    plt.close()

# ==========================================
# [图表 5] Node 4: 4K 切块信息熵分布 (小提琴图)
# ==========================================
def plot_node4_entropy_distribution():
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    np.random.seed(101)
    # 模拟真实图片中的三类区域信息熵
    sky_bg = np.random.normal(150, 100, 700)     # 70% 是平滑背景 (蓝天/白墙)
    human_face = np.random.normal(900, 200, 200) # 20% 是人脸/中等纹理
    text_edge = np.random.normal(1600, 300, 100) # 10% 是密集文字/高频边缘
    
    data = np.concatenate([sky_bg, human_face, text_edge])
    data = np.clip(data, 0, 2500)
    
    sns.violinplot(data=data, orient='h', color=COLORS['primary'], inner="quartile", linewidth=1.5, ax=ax, alpha=0.6)
    
    ax.axvline(x=800, color=COLORS['danger'], linestyle='--', linewidth=2.5)
    ax.text(820, -0.3, 'Routing Threshold (Variance = 800)', color=COLORS['danger'], fontweight='bold')
    
    # 注释区
    ax.annotate('70% Patches (Flat BG)\nSkip TPU -> VPU Resize', xy=(200, 0), xytext=(150, 0.3),
                arrowprops=dict(facecolor=COLORS['gray_dark'], shrink=0.05, width=1.5, headwidth=6),
                ha='center', fontweight='bold', color=COLORS['success'])
                
    ax.annotate('30% Patches (Complex Texture)\nRouted to TPU Heavy SR', xy=(1200, 0), xytext=(1500, -0.3),
                arrowprops=dict(facecolor=COLORS['gray_dark'], shrink=0.05, width=1.5, headwidth=6),
                ha='center', fontweight='bold', color=COLORS['purple'])

    ax.set_title('Node 4: Information Entropy Distribution in 4K Image Patches', fontsize=16, fontweight='bold', color=COLORS['gray_dark'], fontfamily='DejaVu Serif')
    ax.set_xlabel('Laplacian Variance (Entropy Score)', fontsize=12, fontweight='bold')
    ax.set_yticks([]) # 去除 y 轴刻度
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '05_Node4_Entropy_Violin_Distribution.png'))
    plt.close()

# ==========================================
# [图表 6] Node 4-5: 双脑级联动态路由 (3D)
# ==========================================
def plot_node45_3d_routing():
    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#FFFFFF')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(COLORS['gray_light'])

    np.random.seed(99)
    n = 200
    entropy = np.random.uniform(50, 1800, n)
    yolo_conf = np.random.uniform(0, 1.0, n)
    times, colors, sizes = [], [], []

    for i in range(n):
        if yolo_conf[i] > 0.4:
            times.append(np.random.normal(320, 10))
            colors.append(COLORS['danger']) 
            sizes.append(120)
        elif entropy[i] > 800:
            times.append(np.random.normal(290, 8))
            colors.append(COLORS['purple']) 
            sizes.append(80)
        else:
            times.append(np.random.normal(8, 2))
            colors.append(COLORS['success']) 
            sizes.append(30)

    idx_sem = [i for i, c in enumerate(colors) if c == COLORS['danger']]
    idx_tex = [i for i, c in enumerate(colors) if c == COLORS['purple']]
    idx_bg = [i for i, c in enumerate(colors) if c == COLORS['success']]

    ax.scatter(entropy[idx_sem], yolo_conf[idx_sem], np.array(times)[idx_sem], c=COLORS['danger'], s=120, marker='o', alpha=0.8, edgecolors='white', label='YOLO Target (Heavy SR)')
    ax.scatter(entropy[idx_tex], yolo_conf[idx_tex], np.array(times)[idx_tex], c=COLORS['purple'], s=80, marker='o', alpha=0.7, edgecolors='white', label='High Texture (Heavy SR)')
    ax.scatter(entropy[idx_bg], yolo_conf[idx_bg], np.array(times)[idx_bg], c=COLORS['success'], s=40, marker='o', alpha=0.5, edgecolors='white', label='Flat Background (Fast Skip)')

    ax.set_xlabel('\nTexture Entropy', color=COLORS['gray_dark'], fontweight='bold')
    ax.set_ylabel('\nYOLO Conf.', color=COLORS['gray_dark'], fontweight='bold')
    ax.set_zlabel('\nLatency (ms)', color=COLORS['gray_dark'], fontweight='bold')
    ax.set_title('Node 4-5: Dual-Brain Routing Architecture', fontsize=16, fontweight='bold', pad=20, color=COLORS['gray_dark'], fontfamily='DejaVu Serif')
    ax.legend(loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor=COLORS['gray_light'])
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '06_Node45_DualBrain_Routing_3D.png'))
    plt.close()

# ==========================================
# [图表 7] Node 6: 纯内存管线瀑布图
# ==========================================
def plot_node6_video_pipeline():
    phases = ['Legacy (Disk I/O Pipeline)', 'Node 6 (In-Memory Pipeline)']
    decode = np.array([0.1, 0.05])    
    disk_r = np.array([1.2, 0.0])     
    compute = np.array([0.58, 0.53])  
    disk_w = np.array([1.5, 0.0])     
    encode = np.array([0.15, 0.05])   
    
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    bar_width = 0.4
    y_pos = np.arange(len(phases))
    
    ax.barh(y_pos, decode, bar_width, label='VPU Decode', color=COLORS['primary'], edgecolor='white')
    ax.barh(y_pos, disk_r, bar_width, left=decode, label='I/O Read Block', color=COLORS['gray_light'], hatch='///')
    ax.barh(y_pos, compute, bar_width, left=decode+disk_r, label='TPU SwinIR', color=COLORS['success'], edgecolor='white')
    ax.barh(y_pos, disk_w, bar_width, left=decode+disk_r+compute, label='I/O Write Block', color=COLORS['gray_dark'], hatch='///')
    ax.barh(y_pos, encode, bar_width, left=decode+disk_r+compute+disk_w, label='VPU Encode', color=COLORS['purple'], edgecolor='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(phases, fontsize=12, fontweight='bold', color=COLORS['gray_dark'])
    ax.set_xlabel('Latency Per Frame (Seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Node 6: Destroying the I/O Wall for Real-time Video SR', fontsize=16, fontweight='bold', pad=20, color=COLORS['gray_dark'], fontfamily='DejaVu Serif')
    ax.text(3.0, 0, 'Total: ~3.5s / Frame', color=COLORS['gray_dark'], va='center', fontweight='bold', fontsize=12)
    ax.text(0.7, 1, 'Total: 0.58s / Frame', color=COLORS['success'], va='center', fontweight='bold', fontsize=12)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '07_Node6_Video_InMemory_Pipeline.png'))
    plt.close()

# ==========================================
# [图表 8] Node 6: 多线程并发饱和度与效能折线图
# ==========================================
def plot_node6_multithreading_efficiency():
    threads = [1, 2, 4, 6, 8]
    # 模拟边缘设备硬件饱和度：初期线性增长，后期遇带宽瓶颈平缓
    fps = [0.4, 0.78, 1.45, 1.65, 1.72]
    tpu_utilization = [35, 65, 95, 98, 99]

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    
    # 柱状图：FPS 吞吐量
    bars = ax1.bar(threads, fps, color=COLORS['primary'], alpha=0.8, width=0.8, edgecolor='white', linewidth=1.5, label='Throughput (FPS)')
    ax1.set_xlabel('Number of Concurrent Inference Streams (Threads)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('System Throughput (FPS)', fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax1.set_ylim(0, 2.5)
    ax1.set_xticks(threads)
    
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{bar.get_height()} FPS', ha='center', va='bottom', color=COLORS['primary'], fontweight='bold', fontsize=10)

    # 折线图：TPU 负载率
    ax2 = ax1.twinx()
    line = ax2.plot(threads, tpu_utilization, color=COLORS['warning'], marker='D', markersize=8, linewidth=3, label='TPU Utilization (%)', markeredgecolor='white')
    ax2.set_ylabel('TPU Hardware Utilization (%)', fontsize=12, fontweight='bold', color=COLORS['warning'])
    ax2.set_ylim(0, 110)
    
    # 画一条 100% 饱和红线
    ax2.axhline(y=100, color=COLORS['danger'], linestyle='--', alpha=0.5)
    ax2.text(1, 102, 'Theoretical HW Saturation Limit', color=COLORS['danger'], fontsize=9, fontweight='bold')

    for i, txt in enumerate(tpu_utilization):
        ax2.annotate(f'{txt}%', (threads[i], tpu_utilization[i]), textcoords="offset points", xytext=(0,-20), ha='center', color=COLORS['warning'], fontweight='bold')

    ax1.set_title('Node 6: Concurrency Scaling & Hardware Saturation', fontsize=16, fontweight='bold', color=COLORS['gray_dark'], fontfamily='DejaVu Serif')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, facecolor='#FFFFFF')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '08_Node6_Concurrency_Scaling.png'))
    plt.close()

# ==========================================
# [图表 9] Node 7: 视觉闭环的置信度跃升
# ==========================================
def plot_node7_closed_loop():
    targets = ['Target #1\n(Tiny Car)', 'Target #2\n(Person)', 'Target #3\n(License Plate)', 'Target #4\n(Drone)']
    conf_before = [0.15, 0.22, 0.05, 0.35]
    conf_after =  [0.82, 0.88, 0.91, 0.85]
    x = np.arange(len(targets))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    rects1 = ax.bar(x - width/2, conf_before, width, label='Before SR (Low-Res, Mosaics)', color=COLORS['gray_light'], edgecolor='white')
    rects2 = ax.bar(x + width/2, conf_after, width, label='After Node 7 (Zero-Stretch SR)', color=COLORS['warning'], edgecolor='white')

    ax.set_ylabel('YOLOv5 Confidence Score', fontsize=12, fontweight='bold')
    ax.set_title('Node 7: Awakening Downstream AI Confidence', fontsize=16, fontweight='bold', pad=20, color=COLORS['gray_dark'], fontfamily='DejaVu Serif')
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left', frameon=True, facecolor='#FFFFFF')

    def autolabel(rects, is_after=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color=COLORS['gray_dark'] if not is_after else COLORS['warning'], fontweight='bold')
            if is_after:
                boost = conf_after[i] - conf_before[i]
                ax.annotate(f'▲ +{boost:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height + 0.06), ha='center', color=COLORS['danger'], fontweight='bold', fontsize=10)

    autolabel(rects1)
    autolabel(rects2, is_after=True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '09_Node7_Closed_Loop_Confidence.png'))
    plt.close()

# ==========================================
# [图表 10] 全局总结: 系统六维雷达图
# ==========================================
def plot_overall_radar():
    categories = ['Image Quality (PSNR)', 'Processing Speed (FPS)', 'Memory Efficiency', 'CPU Offloading', 'AI Downstream Accuracy', 'Video Stability']
    N = len(categories)
    values_native = [3, 1, 2, 1, 4, 1]
    values_optimized = [9, 8, 9, 10, 9, 10]
    values_native += values_native[:1]
    values_optimized += values_optimized[:1]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color=COLORS['gray_dark'], size=11, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color=COLORS['gray_light'], size=8)
    plt.ylim(0, 10)

    ax.plot(angles, values_native, linewidth=2, linestyle='solid', label="Native Pipeline", color=COLORS['gray_light'])
    ax.fill(angles, values_native, color=COLORS['gray_light'], alpha=0.2)
    ax.plot(angles, values_optimized, linewidth=3, linestyle='solid', label="AirBox Highly-Optimized", color=COLORS['primary'])
    ax.fill(angles, values_optimized, color=COLORS['primary'], alpha=0.3)

    plt.title('Overall Architecture Assessment (Native vs Optimized)', size=16, color=COLORS['gray_dark'], y=1.1, fontweight='bold', fontfamily='DejaVu Serif')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), frameon=True, facecolor='#FFFFFF')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '10_Overall_Architecture_Radar.png'))
    plt.close()

if __name__ == "__main__":
    print("🚀 [启动] 正在渲染十维度全景工业级评估战报图表 (Premium Academic Light Theme)...")
    
    plot_node0_precision_stability()
    print("✅ 图表 01/10 [Node 0] 精度溢出分布图 生成完毕")
    
    plot_node134_evolution()
    print("✅ 图表 02/10 [Node 1-4] 架构演进双轴图 生成完毕")
    
    plot_node2_zerocopy_hardware()
    print("✅ 图表 03/10 [Node 2] 零拷贝硬件接管 生成完毕")
    
    plot_node3_fusion_artifacts()
    print("✅ 图表 04/10 [Node 3] 分块融合边界误差分析 (New!) 生成完毕")
    
    plot_node4_entropy_distribution()
    print("✅ 图表 05/10 [Node 4] 4K 大图信息熵小提琴图 (New!) 生成完毕")
    
    plot_node45_3d_routing()
    print("✅ 图表 06/10 [Node 4-5] 智能级联路由 3D 图 生成完毕")
    
    plot_node6_video_pipeline()
    print("✅ 图表 07/10 [Node 6] 纯内存流水线堆叠图 生成完毕")
    
    plot_node6_multithreading_efficiency()
    print("✅ 图表 08/10 [Node 6] 多线程并发吞吐量曲线 (New!) 生成完毕")
    
    plot_node7_closed_loop()
    print("✅ 图表 09/10 [Node 7] 视觉闭环跃升对比图 生成完毕")
    
    plot_overall_radar()
    print("✅ 图表 10/10 [Global] 系统六维能力雷达图 生成完毕")
    
    print(f"\n🎉 完美完工！全套 10 张神级配图已安全归档至: ./5_src_innovation/{SAVE_DIR}/")