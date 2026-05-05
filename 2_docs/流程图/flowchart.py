from graphviz import Digraph


# ─── 全局色彩系统 ───────────────────────────────────────────────
COLORS = {
    # 节点填充
    "start_fill":   "#E1F5EE",   # 青绿 50
    "start_stroke": "#0F6E56",   # 青绿 600
    "start_font":   "#085041",   # 青绿 800

    "step_fill":    "#EEEDFE",   # 紫色 50
    "step_stroke":  "#534AB7",   # 紫色 600
    "step_font":    "#3C3489",   # 紫色 800

    "check_fill":   "#FAEEDA",   # 琥珀 50
    "check_stroke": "#BA7517",   # 琥珀 600
    "check_font":   "#633806",   # 琥珀 900

    "edge_color":   "#888780",   # 灰色 400
    "edge_label":   "#5F5E5A",   # 灰色 600

    # 执行层（蓝色系）
    "exec_fill":    "#E6F1FB",   # 蓝色 50
    "exec_stroke":  "#185FA5",   # 蓝色 600
    "exec_font":    "#0C447C",   # 蓝色 800

    "tpu_fill":     "#B5D4F4",   # 蓝色 100（核心节点更深）
    "tpu_stroke":   "#0C447C",   # 蓝色 800
    "tpu_font":     "#042C53",   # 蓝色 900

    # 策略层（紫色系，与 step 同系）
    "strategy_fill":   "#EEEDFE",
    "strategy_stroke": "#534AB7",
    "strategy_font":   "#3C3489",

    # 输出层
    "output_fill":  "#E1F5EE",
    "output_stroke":"#0F6E56",
    "output_font":  "#085041",

    "log_fill":     "#FAEEDA",
    "log_stroke":   "#BA7517",
    "log_font":     "#633806",

    # 集群背景
    "cluster_bg":   "#F8F8F7",   # 极淡灰
    "cluster_border":"#D3D1C7",  # 灰色 100
}


def _node_attrs(fill, stroke, font, shape="box", extra=None):
    """构造统一节点属性字典"""
    attrs = dict(
        shape=shape,
        style="rounded,filled",
        fillcolor=fill,
        color=stroke,
        fontcolor=font,
        penwidth="0.8",
    )
    if extra:
        attrs.update(extra)
    return attrs


def create_deployment_pipeline():
    """
    图1：模型部署链路流程图
    PyTorch .pth → ONNX → MLIR → BModel → 端侧验证 → 部署就绪
    """
    dot = Digraph(comment="Model Deployment Pipeline", format="png")

    dot.attr(
        rankdir="TB",
        splines="spline",
        nodesep="0.7",
        ranksep="0.9",
        fontname="Helvetica Neue,Arial,sans-serif",
        bgcolor="white",
        pad="0.4",
    )
    dot.attr("node",
             fontname="Helvetica Neue,Arial,sans-serif",
             fontsize="12",
             margin="0.25,0.15")
    dot.attr("edge",
             fontname="Helvetica Neue,Arial,sans-serif",
             fontsize="10",
             color=COLORS["edge_color"],
             fontcolor=COLORS["edge_label"],
             arrowsize="0.8",
             penwidth="1.0")

    # ── 起止节点 ──
    oval = _node_attrs(
        COLORS["start_fill"], COLORS["start_stroke"], COLORS["start_font"],
        shape="ellipse"
    )
    dot.node("start", "开始\nPyTorch .pth", **oval)
    dot.node("end",   "结束\n部署就绪",     **oval)

    # ── 核心步骤节点 ──
    s = _node_attrs(COLORS["step_fill"], COLORS["step_stroke"], COLORS["step_font"])
    dot.node("step1", "模型导出\nExport to ONNX",               **s)
    dot.node("step2", "格式转换\nONNX → MLIR\n[model_transform.py]", **s)
    dot.node("step3", "编译优化\nMLIR → BModel\n[model_deploy.py]",  **s)
    dot.node("step4", "端侧验证\nValidity & Precision",          **s)

    # ── 检查节点（菱形）──
    c = _node_attrs(
        COLORS["check_fill"], COLORS["check_stroke"], COLORS["check_font"],
        shape="diamond", extra={"margin": "0.3,0.2"}
    )
    dot.node("c1", "算子兼容性检查",      **c)
    dot.node("c2", "显存/算子边界检查",   **c)

    # ── 连接边 ──
    dot.edge("start",  "step1")
    dot.edge("step1",  "c1")
    dot.edge("c1",     "step2", label="通过 ✓")
    dot.edge("step2",  "c2")
    dot.edge("c2",     "step3", label="通过 ✓")
    dot.edge("step3",  "step4")
    dot.edge("step4",  "end",   label="验证通过 ✓")

    dot.render("output_deployment_pipeline", view=False, cleanup=True)
    print("图1 已生成：output_deployment_pipeline.png")


def create_system_architecture():
    """
    图2：系统运行与数据流架构图
    四层协同：输入层 → 策略层 → 执行层 → 输出与评估层
    """
    dot = Digraph(comment="System Architecture", format="png")

    dot.attr(
        rankdir="LR",
        splines="polyline",
        nodesep="0.7",
        ranksep="1.2",
        fontname="Helvetica Neue,Arial,sans-serif",
        bgcolor="white",
        pad="0.4",
    )
    dot.attr("node",
             fontname="Helvetica Neue,Arial,sans-serif",
             fontsize="12",
             style="rounded,filled",
             shape="box",
             margin="0.22,0.14")
    dot.attr("edge",
             fontname="Helvetica Neue,Arial,sans-serif",
             fontsize="10",
             color=COLORS["edge_color"],
             fontcolor=COLORS["edge_label"],
             arrowsize="0.8",
             penwidth="1.0")

    # ── 集群公共样式 ──
    cluster_style = dict(
        style="rounded,filled",
        fillcolor=COLORS["cluster_bg"],
        color=COLORS["cluster_border"],
        penwidth="1.0",
        fontname="Helvetica Neue,Arial,sans-serif",
        fontsize="11",
        fontcolor="#5F5E5A",
        margin="14",
    )

    # ── 1. 输入层 ──
    with dot.subgraph(name="cluster_input") as c:
        c.attr(label="输入层", **cluster_style)
        c.node("input_src",
               "多源输入\n图像 / 视频 / 端云报文",
               fillcolor=COLORS["output_fill"],
               color=COLORS["output_stroke"],
               fontcolor=COLORS["output_font"])

    # ── 2. 策略层 ──
    with dot.subgraph(name="cluster_strategy") as c:
        c.attr(label="策略层", **cluster_style)
        s = dict(fillcolor=COLORS["strategy_fill"],
                 color=COLORS["strategy_stroke"],
                 fontcolor=COLORS["strategy_font"])
        c.node("router",   "智能路由决策\n语义 / 纹理双路由", **s)
        c.node("temporal", "时序复用判定",                    **s)
        c.node("dvfs",     "DVFS 能效调度",                   **s)

    # ── 3. 执行层 ──
    with dot.subgraph(name="cluster_exec") as c:
        c.attr(label="执行层", **cluster_style)
        e = dict(fillcolor=COLORS["exec_fill"],
                 color=COLORS["exec_stroke"],
                 fontcolor=COLORS["exec_font"])
        t = dict(fillcolor=COLORS["tpu_fill"],
                 color=COLORS["tpu_stroke"],
                 fontcolor=COLORS["tpu_font"],
                 penwidth="1.5")
        c.node("pre_proc",  "预处理模块\n硬件接管 / 分块", **e)
        c.node("tpu_core",  "TPU 推理引擎\nSwinIR BModel", **t)
        c.node("post_proc", "后处理模块\n融合 / 格式转换", **e)

    # ── 4. 输出层 ──
    with dot.subgraph(name="cluster_output") as c:
        c.attr(label="输出与评估层", **cluster_style)
        c.node("output",
               "结果输出\n增强图像 / 视频流",
               fillcolor=COLORS["output_fill"],
               color=COLORS["output_stroke"],
               fontcolor=COLORS["output_font"])
        c.node("log",
               "日志闭环\n质量 / 时延 / 功耗",
               fillcolor=COLORS["log_fill"],
               color=COLORS["log_stroke"],
               fontcolor=COLORS["log_font"])

    # ── 数据流边 ──
    # 输入 → 策略
    dot.edge("input_src", "router",   label="接入请求")

    # 策略层内部
    dot.edge("router",   "temporal",  label="视频流")
    dot.edge("temporal", "dvfs",      label="能效模式")

    # 策略 → 执行
    dot.edge("dvfs",   "pre_proc", label="调度指令")
    dot.edge("router", "tpu_core", label="空间分配",
             style="dashed", color="#AFA9EC")

    # 执行层流水线
    dot.edge("pre_proc",  "tpu_core",  label="张量输入")
    dot.edge("tpu_core",  "post_proc", label="推理结果")

    # 执行 → 输出
    dot.edge("post_proc", "output")
    dot.edge("post_proc", "log",    label="旁路监控",
             style="dashed", color="#AFA9EC")

    dot.render("output_system_architecture", view=False, cleanup=True)
    print("图2 已生成：output_system_architecture.png")


if __name__ == "__main__":
    create_deployment_pipeline()
    create_system_architecture()