#!/usr/bin/env python3
"""
Unit Test Suite for Multi-Model Discovery & Visual Camouflage Detection Output
Tests all model files in models/ and validates that visual output masks are generated properly.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_model_inference import (
    discover_all_models,
    find_default_sample_images,
    load_and_preprocess_image,
    ModelRunner,
    process_prediction_mask,
    save_mask_png,
    save_color_heatmap_png,
    save_contour_overlay_image,
    save_multi_model_comparison
)


class TestAllModelsInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project_root = PROJECT_ROOT
        cls.models_dir = cls.project_root / "models"
        cls.output_dir = cls.project_root / "tests" / "unit" / "output"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.sample_images = find_default_sample_images()

    def test_01_model_discovery(self):
        """Verifies that discover_all_models finds models in models/."""
        discovered = discover_all_models(self.models_dir)
        print(f"\n[Test Discovery] Discovered {len(discovered)} model(s) in {self.models_dir}:")
        for m in discovered:
            print(f"  • {m['name']} ({m['model_type'].upper()}, {m['parent_folder']})")
        self.assertGreater(len(discovered), 0, "No model checkpoints found in models/")

    def test_02_inference_all_discovered_models(self):
        """Runs inference for all discovered models on sample image and verifies output masks."""
        discovered = discover_all_models(self.models_dir)
        self.assertGreater(len(discovered), 0, "No models to test.")

        image_path = self.sample_images[0]
        image_stem = Path(image_path).stem
        input_tensor, original_image = load_and_preprocess_image(image_path)

        model_results = []

        for m_info in discovered:
            with self.subTest(model=m_info["name"]):
                runner = ModelRunner(m_info)
                pred_prob = runner.predict(input_tensor)

                self.assertEqual(pred_prob.ndim, 3, "Prediction should be [H, W, C]")
                self.assertEqual(pred_prob.shape[0], 384, "Height should be 384")
                self.assertEqual(pred_prob.shape[1], 384, "Width should be 384")

                # Apply min-max dynamic range sharpening & adaptive thresholding
                scaled_prob, pred_binary = process_prediction_mask(
                    pred_prob,
                    threshold=0.5,
                    sharpen=True,
                    adaptive_threshold=True
                )

                tag = m_info["tag"]
                mask_path = self.output_dir / f"test_{image_stem}_{tag}_pred_mask.png"
                prob_path = self.output_dir / f"test_{image_stem}_{tag}_pred_prob.png"
                contour_path = self.output_dir / f"test_{image_stem}_{tag}_contour.png"

                save_mask_png(pred_binary, str(mask_path))
                save_color_heatmap_png(pred_prob, str(prob_path), sharpen=True)
                save_contour_overlay_image(original_image, pred_binary, str(contour_path))

                self.assertTrue(mask_path.exists(), f"Mask file {mask_path} was not created.")
                self.assertTrue(prob_path.exists(), f"Heatmap file {prob_path} was not created.")
                self.assertTrue(contour_path.exists(), f"Contour overlay file {contour_path} was not created.")

                # Verify that the generated binary mask is non-trivial (contains detected camouflage regions)
                mask_size = mask_path.stat().st_size
                coverage = float(np.mean(pred_binary))
                print(f"    • [{tag}] Coverage: {coverage*100:.2f}% | Mask PNG size: {mask_size} bytes")
                self.assertGreater(mask_size, 500, f"Mask file {mask_path} is too small ({mask_size} bytes).")
                self.assertGreater(coverage, 0.0, f"Binary mask for {m_info['name']} is completely empty (all zeros).")

                model_results.append({
                    "info": m_info,
                    "tag": tag,
                    "binary_mask": pred_binary,
                    "pred_prob": scaled_prob
                })

        # Test multi-model comparison generation
        multi_comp_path = self.output_dir / f"test_{image_stem}_all_models_comparison.png"
        save_multi_model_comparison(original_image, None, model_results, str(multi_comp_path))
        self.assertTrue(multi_comp_path.exists(), "Multi-model comparison image was not created.")


if __name__ == "__main__":
    unittest.main()

