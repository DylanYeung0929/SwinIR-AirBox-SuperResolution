# **SwinIR 全链路部署与演示操作手册 (2026版)**

本操作手册基于项目最终版文件结构编写，涵盖从模型编译到 15 个核心节点的实机演示流程。请按照以下步骤执行测试与答辩演示。

## **一、 项目文件结构总览**

项目资产已归档为 5 个核心目录：

* 1\_data/: 包含校准数据集 (cali\_data) 和图像源文件 (input\_image)。  
* 3\_models/: 包含原始 ONNX 模型及编译后的 TPU .bmodel 二进制文件。  
* 4\_src\_baseline/: 基准测试源码。  
* 5\_src\_innovation/: 包含硬件实测脚本 (tasks) 和数据可视化脚本 (analysis)。

## **二、 模型转换与编译阶段 (宿主机 Docker 环境)**

此阶段在 Ubuntu 宿主机的 Docker 容器内执行，用于将原始模型编译为 BM1684X 指令集。

### **1\. 进入编译工作区**

\# 启动并进入容器 (若未启动)  
docker start swinir\_box && docker exec \-it swinir\_box /bin/bash

\# 定位到模型编译目录  
cd /workspace/3\_models/bmodel/

### **2\. 生成中间表达 (ONNX \-\> MLIR)**

将模型结构转化为 MLIR 中间格式：

model\_transform.py \\  
    \--model\_name SwinIR \\  
    \--model\_def ../onnx/SwinIR-M\_x4.onnx \\  
    \--input\_shapes \[\[1,3,192,192\]\] \\  
    \--mean 0,0,0 \\  
    \--scale 0.00392,0.00392,0.00392 \\  
    \--pixel\_format rgb \\  
    \--mlir SwinIR.mlir

### **3\. 生成硬件二进制与量化模型 (MLIR \-\> BModel)**

**生成 BF16 模型：**

model\_deploy.py \\  
    \--mlir SwinIR.mlir \\  
    \--quantize BF16 \\  
    \--chip bm1684x \\  
    \--model SwinIR\_1684x\_bf16.bmodel

**生成混合精度模型 (Node 8)：**

model\_deploy.py \\  
    \--mlir SwinIR.mlir \\  
    \--quantize INT8 \\  
    \--quantize\_table SwinIR\_mixed\_qtable.ini \\  
    \--chip bm1684x \\  
    \--model SwinIR\_1684x\_mixed.bmodel

## **三、 基础模型与视频流验证 (AirBox 环境)**

此阶段在 AirBox 设备上执行。通过 SSH 登录：ssh linaro@192.168.137.100。

*注：建议开启额外终端执行 bm-smi \-loop 监控硬件状态。*

### **1\. Baseline 基准性能测试 (Node 0\)**

测试系统在基准条件下的并发处理能力：

cd \~/SwinIR\_AirBox\_Project/4\_src\_baseline/

python3 main.py \\  
    \--input\_dir ../1\_data/input\_image \\  
    \--bmodel\_path ../3\_models/bmodel/SwinIR\_1684x\_bf16.bmodel \\  
    \--scale 4 \\  
    \--window\_size 8 \\  
    \--num\_streams 4

### **2\. Node 6: 多线程视频超分处理**

验证 YOLO 与 SwinIR 双流并发管线：

cd \~/SwinIR\_AirBox\_Project/5\_src\_innovation/tasks/

python3 node6\_video\_main.py \\  
    \--video\_path ../../test\_traffic.mp4 \\  
    \--bmodel\_path ../../3\_models/bmodel/SwinIR\_1684x\_bf16.bmodel \\  
    \--yolo\_bmodel ../../3\_models/bmodel/yolov5s/yolov5s.bmodel \\  
    \--num\_streams 4 \\  
    \--scale 4 \\  
    \--tile 192

## **四、 核心节点实机演示 (AirBox 环境)**

本环节为主要演示内容。执行路径统一为：cd \~/SwinIR\_AirBox\_Project/5\_src\_innovation/tasks/

### **Node 7: 目标检测闭环验证**

验证超分处理对下游目标检测任务的准确率提升：

python3 node7\_detect\_loop.py \\  
    \--img\_path ../../1\_data/cali\_data/detection\_test.jpg \\  
    \--bmodel\_path ../../3\_models/bmodel/SwinIR\_1684x\_bf16.bmodel \\  
    \--yolo\_bmodel ../../3\_models/bmodel/yolov5s/yolov5s.bmodel

### **Node 10: 端云协同传输验证**

测试高压缩比图像在边缘侧的 TPU 重建效果：

python3 node10\_real\_deploy.py \\  
    \--img\_path ../../1\_data/cali\_data/detection\_test.jpg \\  
    \--bmodel\_path ../../3\_models/bmodel/SwinIR\_1684x\_bf16.bmodel \\  
    \--quality 20

### **Node 11: 综合视频超分引擎**

结合感知路由与时序特征复用，导出 4K 对比视频：

python3 node11\_ultimate\_real\_video\_sr.py \\  
    \--video\_path ../../test\_traffic.mp4 \\  
    \--bmodel\_path ../../3\_models/bmodel/SwinIR\_1684x\_bf16.bmodel \\  
    \--yolo\_bmodel ../../3\_models/bmodel/yolov5s/yolov5s.bmodel \\  
    \--output\_path ./ultimate\_showcase.mp4

### **Node 12: C++ 底层算子加速验证**

对比 C++ 算子与 Python 环境的执行效率：

\# 验证算子提速比  
python3 node12\_tpu\_op\_deploy.py

\# 验证内存指针操作耗时  
python3 node12\_real\_tpu\_op.py

### **Node 13: DVFS 动态功耗调度验证**

生成包含硬件 OSD 信息的视频，验证系统动态能效管理：

python3 node13\_real\_video\_dvfs.py \\  
    \--video\_path ../../test\_traffic.mp4

### **Node 15: 模型蒸馏与动态调度**

验证系统在高负载条件下的模型自动切换机制：

python3 node15\_real\_distillation\_routing.py \\  
    \--teacher ../../3\_models/bmodel/SwinIR\_1684x\_bf16.bmodel \\  
    \--student ../../3\_models/bmodel/SwinIR\_1684x\_mixed.bmodel \\  
    \--image ../../1\_data/cali\_data/detection\_test.jpg

