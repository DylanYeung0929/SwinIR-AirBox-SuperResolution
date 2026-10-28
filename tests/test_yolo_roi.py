"""Verify YOLO output coordinates without requiring TPU hardware or OpenCV."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


class YoloRoiTests(unittest.TestCase):
    def test_anchor_row_column_map_to_the_correct_region(self):
        cv2 = types.ModuleType("cv2")
        cv2.COLOR_BGR2RGB = 1
        cv2.resize = lambda image, size: np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.cvtColor = lambda image, mode: image
        sophon = types.ModuleType("sophon")
        sail = types.ModuleType("sophon.sail")
        sophon.sail = sail
        utils = types.ModuleType("utils")
        utils.calculate_psnr = lambda *args: 0
        inference = types.ModuleType("inference")
        inference.SwinIRInference = object

        modules = {"cv2": cv2, "sophon": sophon, "sophon.sail": sail,
                   "utils": utils, "inference": inference}
        pipeline_path = Path(__file__).resolve().parents[1] / "4_src_baseline" / "pipeline.py"
        spec = importlib.util.spec_from_file_location("baseline_pipeline", pipeline_path)
        pipeline = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(pipeline)

        logits = np.full((1, 3, 5, 10, 85), -20.0, dtype=np.float32)
        logits[0, 1, 2, 7, 4] = 10.0  # anchor 1, row 2, column 7
        engine = types.SimpleNamespace(process=lambda graph, data: {"output": logits})
        detector = pipeline.YOLOv5_ROI.__new__(pipeline.YOLOv5_ROI)
        detector.engine = engine
        detector.graph_name = "graph"
        detector.input_name = "input"
        detector.output_names = ["output"]
        detector.input_shape = (1, 3, 20, 20)

        mask = detector.get_roi_mask(np.zeros((100, 200, 3), dtype=np.uint8))
        self.assertEqual(mask[40, 140], 255)  # column 7 / 10, row 2 / 5
        self.assertEqual(mask[20, 0], 0)


if __name__ == "__main__":
    unittest.main()
