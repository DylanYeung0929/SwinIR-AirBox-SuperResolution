"""Test baseline CLI defaults without importing the hardware runtime."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


config_path = Path(__file__).resolve().parents[1] / "4_src_baseline" / "config.py"
spec = importlib.util.spec_from_file_location("baseline_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


class BaselineConfigTests(unittest.TestCase):
    def parse(self, *extra):
        argv = ["main.py", "--input_dir", "image.png", "--bmodel_path", "model.bmodel",
                "--scale", "4", "--window_size", "8", *extra]
        with patch.object(sys, "argv", argv):
            return config.get_args()

    def test_default_tile_produces_a_positive_stride(self):
        args = self.parse()
        self.assertEqual(args.tile, 192)
        self.assertGreater(args.tile - args.tile_overlap, 0)

    def test_explicit_tile_remains_configurable(self):
        args = self.parse("--tile", "256", "--tile_overlap", "32")
        self.assertEqual(args.tile, 256)
        self.assertEqual(args.tile - args.tile_overlap, 224)


if __name__ == "__main__":
    unittest.main()
