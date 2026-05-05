# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository scope

This repository combines three related code areas:

- `4_src_baseline/`: baseline SwinIR TPU deployment pipeline (CLI + Flask service).
- `5_src_innovation/`: node-based innovation experiments (Node 6–15), plus shared `core/` inference utilities and `analysis/` chart/report scripts.
- `CICC0902071-初赛-技术代码/CICC0902071-初赛-技术代码/SwinIR/`: upstream/reference SwinIR PyTorch code (`main_test_swinir.py`, model defs, utility scripts).

There is no single repo-wide Python package, no `pyproject.toml`, and no discovered top-level lint/test automation config.

## Common commands

Run commands from repository root unless otherwise noted.

### Baseline TPU inference (image/file/dir input)

```bash
python 4_src_baseline/main.py \
  --input_dir 1_data/cali_data \
  --bmodel_path 3_models/bmodel/SwinIR_1684x_bf16.bmodel \
  --scale 4 \
  --window_size 8
```

### Baseline Flask service

```bash
python 4_src_baseline/app.py \
  --input_dir 1_data/cali_data \
  --bmodel_path 3_models/bmodel/SwinIR_1684x_bf16.bmodel \
  --scale 4 \
  --window_size 8
```

### Upstream SwinIR reference test/eval (single task)

```bash
python CICC0902071-初赛-技术代码/CICC0902071-初赛-技术代码/SwinIR/main_test_swinir.py \
  --task classical_sr \
  --scale 4 \
  --training_patch_size 48 \
  --model_path CICC0902071-初赛-技术代码/CICC0902071-初赛-技术代码/SwinIR/model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth \
  --folder_lq CICC0902071-初赛-技术代码/CICC0902071-初赛-技术代码/SwinIR/testsets/Set5/LR_bicubic/X4 \
  --folder_gt CICC0902071-初赛-技术代码/CICC0902071-初赛-技术代码/SwinIR/testsets/Set5/HR
```

### Innovation nodes (single script runs)

Node scripts are in `5_src_innovation/tasks/` and are intended to be run individually.

Examples:

```bash
python 5_src_innovation/tasks/node6_video_main.py \
  --video_path test_traffic.mp4 \
  --bmodel_path 3_models/bmodel/SwinIR_1684x_bf16.bmodel
```

```bash
python 5_src_innovation/tasks/node10_real_deploy.py \
  --img_path 1_data/cali_data/detection_test.jpg \
  --bmodel_path 3_models/bmodel/SwinIR_1684x_bf16.bmodel
```

```bash
python 5_src_innovation/tasks/node11_ultimate_real_video_sr.py \
  --video_path test_traffic.mp4 \
  --bmodel_path 3_models/bmodel/SwinIR_1684x_bf16.bmodel
```

```bash
python 5_src_innovation/tasks/node13_real_video_dvfs.py \
  --video_path test_traffic.mp4
```

```bash
python 5_src_innovation/tasks/node15_real_distillation_routing.py \
  --image 1_data/cali_data/detection_test.jpg
```

### Analysis/chart generation

```bash
python 5_src_innovation/analysis/master_report_charts.py
python 5_src_innovation/analysis/node10_analysis.py
python 5_src_innovation/analysis/node11_analysis.py
python 5_src_innovation/analysis/node12_analysis.py
python 5_src_innovation/analysis/node13_analysis.py
python 5_src_innovation/analysis/node14_analysis.py
python 5_src_innovation/analysis/node15_analysis.py
```

## Build/lint/test status

- No repo-wide lint command was found (no discovered `ruff`, `flake8`, `pylint`, or similar config at root).
- No repo-wide automated unit-test suite entrypoint was found (`pytest`/`unittest` workflows are not configured at root).
- Validation is primarily script-based (run the relevant node script directly).

## Architecture quick map

### 1) Baseline runtime path (`4_src_baseline`)

- `config.py`: CLI argument contract shared by CLI and Flask entrypoints.
- `main.py`: batch/image-file driver; builds `Pipeline` and reports timing/PSNR summary.
- `app.py`: Flask wrapper over the same `Pipeline` for upload/status/result APIs.
- `pipeline.py`: core scheduler. Splits images into tiles, performs routing and blending, manages worker threads/queues.
- `inference.py`: `SwinIRInference` wrapper around `sophon.sail.Engine.process`.
- `utils.py`: preprocess/postprocess and PSNR helpers; includes BMCV/SAIL image-tensor conversion logic.

### 2) Innovation stack (`5_src_innovation`)

- `core/`: reusable inference/util logic (same role as baseline inference/utils, used by several node scripts).
- `tasks/`: Node-specific experiments and deployments (Node 6–15). Most are standalone runnable scripts with their own argparse.
- `analysis/`: post-run analysis/plot scripts producing report charts.
- `SwinIR_Implementation_Audit.md` and `inno_logs.txt`: project-level audit/record artifacts.

Key idea across nodes: start from SwinIR TPU inference, then layer routing/scheduling strategies (YOLO ROI semantic routing, entropy-based routing, temporal reuse, DVFS simulation/analysis, edge-cloud bandwidth scenarios, teacher/student switching).

### 3) Reference SwinIR source (`.../SwinIR`)

- `main_test_swinir.py`: upstream-style test/eval entrypoint across SR/denoise/JPEG tasks.
- `network_swinir.py` and `models/network_swinir.py`: model architecture definitions.
- `README.md`: canonical upstream usage examples.

## External/runtime assumptions

- Many scripts require `sophon.sail` runtime and valid `.bmodel` files (BM1684X-oriented paths are common).
- Several innovation scripts also expect a YOLO bmodel path for ROI-based routing.
- `node12_*` scripts reference a prebuilt native library at `/home/linaro/libtpu_op.so`; source file is `5_src_innovation/tasks/tpu_fusion_kernel.cpp`.
- Data/model paths are often hardcoded or default-relative; verify arguments before running.
