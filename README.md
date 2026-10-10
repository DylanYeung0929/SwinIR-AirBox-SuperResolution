<div align="center">

# SwinIR on AirBox

### 图像超分辨率 · 神经网络部署 · 边缘 AI

将 SwinIR 超分辨率推理管线与实验脚本整理到算能 AirBox / BM1684X 平台。  
An edge AI project for SwinIR image super-resolution on SOPHON hardware.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![SwinIR](https://img.shields.io/badge/Model-SwinIR-7C3AED?style=flat-square)
![BM1684X](https://img.shields.io/badge/Target-BM1684X-0F766E?style=flat-square)

[项目结构](#项目结构) · [运行准备](#运行准备) · [实验入口](#实验入口) · [完整操作手册](<2_docs/SwinIR 全链路部署与演示操作手册.md>)

</div>

---

<div align="center">

![图像超分辨率对比演示](<2_docs/演示动图/03_图像超分对比演示.gif>)

<sub>仓库内的图像超分辨率对比演示</sub>

</div>

## 项目概览

本仓库围绕 SwinIR 图像超分辨率的边缘部署，保存了多流推理基线、视频处理与下游检测实验、调度和能效分析脚本、可视化结果，以及比赛文档。研究对象是图像重建神经网络在资源受限硬件上的实际处理流程。

```text
输入图像 / 视频
       ↓
预处理与分块 → SwinIR BModel 推理 → 后处理与拼接
       ↓
结果保存、性能记录与实验分析
```

仓库同时包含硬件运行脚本和仿真实验脚本。具体结果依赖模型版本、输入、SDK 与设备环境；此 README 不把仿真数值当作实机性能。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| [`1_data/`](1_data/) | 示例输入与校准相关数据 |
| [`2_docs/`](2_docs/) | 操作手册、报告、流程图和演示动图 |
| [`4_src_baseline/`](4_src_baseline/) | 基础多流推理、预处理与结果保存 |
| [`5_src_innovation/tasks/`](5_src_innovation/tasks/) | 视频超分、检测闭环、调度与算子实验脚本 |
| [`5_src_innovation/analysis/`](5_src_innovation/analysis/) | 实验数据分析与图表生成脚本 |
| [`results/`](results/) | 仓库中保留的示例结果 |

关键入口包括 [`4_src_baseline/main.py`](4_src_baseline/main.py)、[`4_src_baseline/config.py`](4_src_baseline/config.py)、[`5_src_innovation/tasks/node6_video_main.py`](5_src_innovation/tasks/node6_video_main.py) 和 [`5_src_innovation/tasks/node13_real_video_dvfs.py`](5_src_innovation/tasks/node13_real_video_dvfs.py)。

## 运行准备

1. 准备与目标设备匹配的算能 AirBox / BM1684X 环境、Sophon SDK / `sophon.sail`，以及脚本使用的 Python 图像处理依赖。
2. 准备输入图像或视频，以及与推理参数一致的 SwinIR `.bmodel`。模型转换步骤见[完整操作手册](<2_docs/SwinIR 全链路部署与演示操作手册.md>)。
3. 检查模型输入形状、超分倍数、颜色顺序与归一化设置，再运行基线。

**模型文件不在本仓库中。** 根目录的 `3_models/`、`.bmodel` 和部分大体积权重被 [`.gitignore`](.gitignore) 排除。克隆仓库后不能仅凭源码直接完成硬件推理；需要把相应模型放到本机并传入实际路径。

## 实验入口

基础图像推理示例（从仓库根目录执行，替换尖括号中的路径）：

```bash
python3 4_src_baseline/main.py \
  --input_dir <image-or-directory> \
  --bmodel_path <absolute-path-to-swinir.bmodel> \
  --scale 4 \
  --window_size 8 \
  --num_streams 4
```

| 方向 | 入口 |
| --- | --- |
| 视频超分与并发 | [`node6_video_main.py`](5_src_innovation/tasks/node6_video_main.py) |
| 超分与目标检测 | [`node7_detect_loop.py`](5_src_innovation/tasks/node7_detect_loop.py) |
| 带宽与端侧重建 | [`node10_bandwidth_sim.py`](5_src_innovation/tasks/node10_bandwidth_sim.py)、[`node10_real_deploy.py`](5_src_innovation/tasks/node10_real_deploy.py) |
| 视频处理与时序实验 | [`node11_temporal_reuse_sim.py`](5_src_innovation/tasks/node11_temporal_reuse_sim.py)、[`node11_ultimate_real_video_sr.py`](5_src_innovation/tasks/node11_ultimate_real_video_sr.py) |
| 功耗档位调度 | [`node13_dvfs_sim.py`](5_src_innovation/tasks/node13_dvfs_sim.py)、[`node13_real_video_dvfs.py`](5_src_innovation/tasks/node13_real_video_dvfs.py) |

各脚本的输入文件、模型路径与硬件要求不同。运行前先查看脚本参数，并按[操作手册](<2_docs/SwinIR 全链路部署与演示操作手册.md>)选择对应流程。

## 文档与来源

- [全链路部署与演示操作手册](<2_docs/SwinIR 全链路部署与演示操作手册.md>)：模型转换、设备推理与实验演示的步骤。
- [比赛报告源文件](2_docs/contest_report.tex)：项目报告的 LaTeX 源文件。
- 仓库内的第三方 SwinIR 代码保留其原有说明与许可证；使用时请遵守相应上游许可。
