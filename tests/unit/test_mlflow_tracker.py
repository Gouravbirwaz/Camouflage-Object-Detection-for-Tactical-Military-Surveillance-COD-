import unittest

from src.tracking.mlflow_utils import MLflowTracker


class TestMLflowTrackerCompatibility(unittest.TestCase):
    def test_init_accepts_legacy_log_dir_arg(self):
        tracker = MLflowTracker(experiment_name="compat-test", log_dir="metrics", enabled=False)
        self.assertEqual(tracker.experiment_name, "compat-test")
        self.assertFalse(tracker.enabled)


if __name__ == "__main__":
    unittest.main()
