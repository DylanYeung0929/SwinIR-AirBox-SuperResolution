# **📊 SwinIR 部署创新项目：全量资产“虚实”审计报告 (28 核心文件)**

本报告为 2026 集创赛“算能杯”最终版资产清单。用于 claudecode 上下文注入，确保 AI 能够理解从底层 C++ 到上层 Web 的完整逻辑。

## **🟣 第一类：算法源头与量化资产 (Model & Quant Assets)**

这是项目的“灵魂”，涵盖了从源码到精细化量化配置的全过程。

| 节点/组件 | 对应文件名 | 属性 | 功能描述 |
| :---- | :---- | :---- | :---- |
| **Node 0** | network\_swinir.py | 源码 | SwinIR 原始 PyTorch 定义，是模型转换的起点。 |
| **Node 8/9** | SwinIR\_mixed\_qtable.ini | 配置文件 | **混合精度地图**：手动标定的 BF16 保护层名单。 |
| **Node 12** | tpu\_fusion\_kernel.cpp | **C++ 源码** | **硬件加速核心**：Node 12 的底层实现，用于编译 .so。 |

## **🟩 第二类：物理级真实部署 (Real Hardware Practice)**

这些脚本物理加载了 BModel，并物理调用了 Sophgo BM1684X 的 TPU 算力进行推理。

| 节点/组件 | 对应文件名 | 物理执行逻辑 | 硬件调用证据 |
| :---- | :---- | :---- | :---- |
| **Node 6** | node6\_video\_main.py | 多路并发 | sail.Engine 实例化列表 |
| **Node 7** | node7\_detect\_loop.py | AI 任务闭环 | YOLO 与 SwinIR 联合物理运行 |
| **Node 10** | node10\_real\_deploy.py | 端云重构 | 真实处理压缩字节流并重构 |
| **Node 11** | node11\_real\_deploy.py | 显存复用 | 基于 VRAM 地址的缓存跳过逻辑 |
| **Ultimate** | ultimate\_real\_video\_sr.py | 全能引擎 | 物理级集成感知路由、复用与分流 |
| **Node 12** | node12\_real\_tpu\_op.py | 算子点火 | 通过 ctypes 物理调用底层 .so |
| **Node 12** | node12\_tpu\_op\_deploy.py | 算子部署 | 演示自定义算子在 Pipeline 中的挂载 |
| **Node 15** | node15\_real\_distillation\_routing.py | 蒸馏热切换 | 内存中同时常驻两个不同规格的模型引擎 |

## **🟥 第三类：算法逻辑仿真 (Scientific Simulation)**

用于验证调度决策逻辑的“数学正确性”，是算法落地前的标定基准。

| 节点/组件 | 对应文件名 | 模拟内容 | 核心价值 |
| :---- | :---- | :---- | :---- |
| **Node 8** | node8\_tta\_exp.py | TTA 延迟仿真 | 确定“纹理复杂度”决策阈值 |
| **Node 10** | node10\_bandwidth\_sim.py | 带宽压缩仿真 | 验证“端云协同”的带宽节省率上限 |
| **Node 11** | node11\_temporal\_reuse\_sim.py | 时序复用仿真 | 建立 FPS 提升与复用率的数学模型 |
| **Node 13** | node13\_dvfs\_sim.py | 硬件功耗模型 | 基于 ![][image1] 公式进行能效分析 |
| **Node 14** | node14\_blind\_sr\_perception.py | 退化感知大脑 | 0.3ms 级 CPU 算子感知脏数据逻辑 |
| **Node 15** | node15\_distillation\_engine.py | 蒸馏路由逻辑 | 模拟柔性算力在高负载下的流转响应 |

## **🟦 第四类：系统架构与基础设施 (Infrastructure)**

构成边缘端 SwinIR 重构服务器的物理骨架。

| 组件 | 对应文件名 | 属性 | 功能描述 |
| :---- | :---- | :---- | :---- |
| **Web App** | app.py | 真实 | 基于 Flask 的微服务接口，管理任务生命周期。 |
| **Pipeline** | pipeline.py | 真实 | 核心调度基类，处理分块融合与多流同步。 |
| **Inference** | inference.py | 真实 | **硬件抽象层 (HAL)**：封装了所有的推理细节。 |
| **Hardware** | utils.py | 真实 | **物理加速器**：包含基于 BMCV 硬件 VPP 的预处理。 |
| **Simulator** | VideoSRSimulator.jsx | 模拟 | 前端算力调度看板，用于直观展示架构逻辑。 |

## **🟧 第五类：性能分析与可视化工具 (Tooling & Records)**

用于产出数据战报，是答辩 PPT 图片的直接来源。

| 组件 | 对应文件名 | 属性 | 功能描述 |
| :---- | :---- | :---- | :---- |
| **Profiler** | node12\_profiler.py | 真实 | 物理跟踪函数执行，定位 I/O 墙的具体位置。 |
| **Charts** | innovation\_report\_charts.py | 工具 | 生成 Node 0-11 的所有高级学术对比图表。 |
| **Visualizer** | node13\_real\_video\_dvfs\_visual.py | 工具 | 将实机 DVFS 调度过程渲染为带 OSD 数据的视频。 |
| **Logs** | inno\_logs.txt | 记录 | 记录了 15 个节点所有的物理实测数据点。 |

## **💡 给 Claudecode 的特别指令**

当后续使用 claudecode 进行辅助开发时，请特别关注：

1. **Node 12** 的 C++ 源码与 Python ctypes 的**指针寻址对齐**。  
2. **Node 11** 的 VRAM 缓存与 inference.py 中 engine.process 的**执行顺序**。  
3. **Node 15** 的模型切换逻辑与 **AirBox 物理显存 (950MB)** 的安全水位线。

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEIAAAAWCAYAAAB0S0oJAAABaElEQVR4Xu2UsUoEMRCGTwULKyuRXdm9ZRvdwlatLXwCESzlmkPEygew8TEs1EJEUPAJBLEUxN5CK0EshKuu0H9gAvE3Od0DMezlg+H28ieT+Wc3abUikchP5Hl+jvgYFFmWnfK6xmJM83gbaDMuWWskavaaxwVfkxoHTG6KUbz8NdaSJJkapUY8+IyiOWfaiBXWXGD+MuZeILZZE6AvyZ2D2GHt3/G98aIoFvVL2WfNBebeI65g8sDkxNppS79DPJdlOWOvCwZTtCNOqqqa5PkuYH4VpjfsMazvSx48TuD3BnFs60EBA+tSLEx0WauDGv6GNrSHeGMtKPIB90MdfDnM8ULssRYUWqTTRB18OeRoWXuMsR4MWuQjj9cFOQ55TJD8OHazuk+f9SDA/bCrBW6xNgzI92Se0zSdE+OIjvxHM+Z1r9ffXsB/Doo50qK+BM8bgnHkebdyLjj0F9VuSYtEIpFIJGA+AZSCfxm2u6TeAAAAAElFTkSuQmCC>