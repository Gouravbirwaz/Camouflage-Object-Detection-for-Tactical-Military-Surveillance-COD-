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
    find_ground_truth_for_image,
    load_and_preprocess_image,
    load_ground_truth,
    ModelRunner,
    process_prediction_mask,
    save_mask_png,
    save_comparison_image,
    save_multi_model_comparison
)
from src.models.dgnet import DGNetLoss


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

    def test_02_dgnet_loss_constructor_compatibility(self):
        """Ensures the notebook's legacy DGNetLoss arguments remain supported."""
        loss = DGNetLoss(grad_loss_weight=1.0, deep_supervision=True)
        self.assertIsInstance(loss, tf.keras.losses.Loss)
        self.assertEqual(loss.lambda_grad_loss, 1.0)
        self.assertTrue(loss.deep_supervision)

    def test_03_inference_all_discovered_models(self):
        """Runs inference for all discovered models on sample image and verifies output masks."""
        discovered = discover_all_models(self.models_dir)
        self.assertGreater(len(discovered), 0, "No models to test.")

        image_path = self.sample_images[0]
        image_stem = Path(image_path).stem
        input_tensor, original_image = load_and_preprocess_image(image_path)
        gt_path = find_ground_truth_for_image(image_path)
        gt_mask = load_ground_truth(gt_path) if gt_path else None

        if gt_mask is not None:
            gt_out = self.output_dir / f"test_{image_stem}_gt.png"
            save_mask_png(gt_mask, str(gt_out))
            self.assertTrue(gt_out.exists(), f"GT mask file {gt_out} was not created.")

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
                mask_path = self.output_dir / f"test_{image_stem}_{tag}_pred.png"
                comp_path = self.output_dir / f"test_{image_stem}_{tag}_comparison.png"

                save_mask_png(pred_binary, str(mask_path))
                save_comparison_image(original_image, pred_binary, str(comp_path),
                                      gt_mask=(gt_mask.numpy() if gt_mask is not None else None),
                                      model_name=tag)

                self.assertTrue(mask_path.exists(), f"Mask file {mask_path} was not created.")
                self.assertTrue(comp_path.exists(), f"Comparison file {comp_path} was not created.")

                # Verify that the generated binary mask is valid (0 and 255)
                mask_size = mask_path.stat().st_size
                coverage = float(np.mean(pred_binary))
                print(f"    • [{tag}] Coverage: {coverage*100:.2f}% | Mask PNG size: {mask_size} bytes")
                self.assertGreater(mask_size, 500, f"Mask file {mask_path} is too small ({mask_size} bytes).")

                model_results.append({
                    "info": m_info,
                    "tag": tag,
                    "binary_mask": pred_binary,
                    "pred_prob": scaled_prob
                })

        # Test multi-model comparison generation
        multi_comp_path = self.output_dir / f"test_{image_stem}_all_models_comparison.png"
        save_multi_model_comparison(original_image, gt_mask, model_results, str(multi_comp_path))
        self.assertTrue(multi_comp_path.exists(), "Multi-model comparison image was not created.")


if __name__ == "__main__":
    unittest.main()

